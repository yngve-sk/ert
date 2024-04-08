import os
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field

from ert.config.parsing import ConfigWarning, HistorySource
from ert.config.parsing.observations_parser import (
    DateValues,
    ErrorValues,
    GenObsValues,
    HistoryValues,
    ObservationConfigError,
    SummaryValues,
)
from ert.config.responses.gen_data_config import GenDataConfig
from ert.config.responses.general_observation import GenObservation
from ert.config.responses.summary_observation import SummaryObservation
from ert.validation import rangestring_to_list

from .observation_vector import ObsVector
from .response_properties import ResponseTypes

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config.ensemble_config import EnsembleConfig

DEFAULT_TIME_DELTA = timedelta(seconds=30)


def history_key(key: str) -> str:
    keyword, *rest = key.split(":")
    return ":".join([keyword + "H"] + rest)


class _AccumulatedDataset(Protocol):
    def to_xarray(self) -> xr.Dataset: ...
    def __len__(self) -> int: ...


class _SummaryObsDataset(BaseModel):
    summary_keys: List[str] = Field(default_factory=lambda: [])
    observations: List[float] = Field(default_factory=lambda: [])
    stds: List[float] = Field(default_factory=lambda: [])
    times: List[int] = Field(default_factory=lambda: [])
    obs_names: List[str] = Field(default_factory=lambda: [])

    def __len__(self) -> int:
        return len(self.summary_keys)

    def to_xarray(self) -> xr.Dataset:
        return (
            pd.DataFrame(
                data={
                    "name": self.summary_keys,
                    "obs_name": self.obs_names,
                    "time": self.times,
                    "observations": self.observations,
                    "std": self.stds,
                },
            )
            .set_index(["name", "obs_name", "time"])
            .to_xarray()
        )


class _GenObsDataset(BaseModel):
    gen_data_keys: List[str] = Field(default_factory=lambda: [])
    observations: List[float] = Field(default_factory=lambda: [])
    stds: List[float] = Field(default_factory=lambda: [])
    indexes: List[int] = Field(default_factory=lambda: [])
    report_steps: List[int] = Field(default_factory=lambda: [])
    obs_names: List[str] = Field(default_factory=lambda: [])

    def __len__(self) -> int:
        return len(self.gen_data_keys)

    def to_xarray(self) -> xr.Dataset:
        return (
            pd.DataFrame(
                data={
                    "name": self.gen_data_keys,
                    "obs_name": self.obs_names,
                    "report_step": self.report_steps,
                    "index": self.indexes,
                    "observations": self.observations,
                    "std": self.stds,
                }
            )
            .set_index(["name", "obs_name", "report_step", "index"])
            .to_xarray()
        )


class _GenObsAccumulator:
    def __init__(self):
        self.ds: _GenObsDataset = _GenObsDataset()

    def write(
        self,
        gen_data_key: str,
        obs_name: str,
        report_step: int,
        observations,
        stds,
        indexes: List[int],
    ):
        self.ds.gen_data_keys.extend([gen_data_key] * len(observations))
        self.ds.obs_names.extend([obs_name] * len(observations))
        self.ds.report_steps.extend([report_step] * len(observations))

        self.ds.observations.extend(observations)
        self.ds.stds.extend(stds)
        self.ds.indexes.extend(indexes)


class _SummaryObsAccumulator:
    def __init__(self):
        self.ds: _SummaryObsDataset = _SummaryObsDataset()

    def write(
        self,
        summary_key: str,
        obs_names: List[str],
        observations: List[float],
        stds: List[float],
        times: List[int],
    ):
        self.ds.summary_keys.extend([summary_key] * len(obs_names))
        self.ds.obs_names.extend(obs_names)
        self.ds.observations.extend(observations)
        self.ds.stds.extend(stds)
        self.ds.times.extend(times)

    def to_xarrays_grouped_by_response(self) -> Dict[str, xr.Dataset]:
        return self.ds.to_xarray()


# Columns used to form a key for observations of the response_type
ObservationsIndices = {"summary": ["time"], "gen_data": ["index", "report_step"]}


class EnkfObs:
    def __init__(self, obs_vectors: Dict[str, ObsVector], obs_time: List[datetime]):
        self.obs_vectors = obs_vectors
        self.obs_time = obs_time

        vecs: List[ObsVector] = [*self.obs_vectors.values()]

        gen_obs = _GenObsAccumulator()
        sum_obs = _SummaryObsAccumulator()

        # Faster to not create a single xr.Dataset per
        # observation and then merge/concat
        # this just accumulates 1d vecs before making a dataset
        for vec in vecs:
            if vec.observation_type == ResponseTypes.GEN_DATA:
                for report_step, node in vec.observations.items():
                    gen_obs.write(
                        gen_data_key=vec.response_name,
                        obs_name=vec.observation_name,
                        report_step=report_step,
                        observations=node.values,
                        stds=node.stds,
                        indexes=node.indices,
                    )

            elif vec.observation_type == ResponseTypes.SUMMARY:
                observations = []
                stds = []
                dates = []
                obs_keys = []

                for the_date, obs in vec.observations.items():
                    assert isinstance(obs, SummaryObservation)
                    observations.append(obs.value)
                    stds.append(obs.std)
                    dates.append(the_date)
                    obs_keys.append(obs.observation_key)

                sum_obs.write(
                    summary_key=vec.observation_name,
                    obs_names=obs_keys,
                    observations=observations,
                    stds=stds,
                    times=dates,
                )
            else:
                raise ValueError("Unknown observation type")

        obs_vectors: List[
            Tuple[Literal["gen_data", "summary"], _AccumulatedDataset]
        ] = [
            ("gen_data", gen_obs.ds),
            ("summary", sum_obs.ds),
        ]
        obs_dict: Dict[str, xr.Dataset] = {}
        for key, vec in obs_vectors:
            if len(vec) > 0:
                ds = vec.to_xarray()
                ds.attrs["response"] = key
                obs_dict[key] = ds

        self.datasets: Dict[str, xr.Dataset] = obs_dict

    def __len__(self) -> int:
        return len(self.obs_vectors)

    def __contains__(self, key: str) -> bool:
        return key in self.obs_vectors

    def __iter__(self) -> Iterator[ObsVector]:
        return iter(self.obs_vectors.values())

    def __getitem__(self, key: str) -> ObsVector:
        return self.obs_vectors[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnkfObs):
            return False
        # Datasets contains the full observations, so if they are equal, everything is
        return self.datasets == other.datasets

    def getTypedKeylist(
        self, observation_implementation_type: ResponseTypes
    ) -> List[str]:
        return sorted(
            [
                key
                for key, obs in self.obs_vectors.items()
                if observation_implementation_type == obs.observation_type
            ]
        )

    @staticmethod
    def _handle_error_mode(
        values: "npt.ArrayLike",
        error_dict: ErrorValues,
    ) -> "npt.NDArray[np.double]":
        values = np.asarray(values)
        error_mode = error_dict.error_mode
        error_min = error_dict.error_min
        error = error_dict.error
        if error_mode == "ABS":
            return np.full(values.shape, error)
        elif error_mode == "REL":
            return np.abs(values) * error
        elif error_mode == "RELMIN":
            return np.maximum(np.abs(values) * error, np.full(values.shape, error_min))
        raise ObservationConfigError(f"Unknown error mode {error_mode}", error_mode)

    @classmethod
    def _handle_history_observation(
        cls,
        ensemble_config: "EnsembleConfig",
        history_observation: HistoryValues,
        summary_key: str,
        std_cutoff: float,
        history_type: HistorySource,
        time_len: int,
    ) -> Dict[str, ObsVector]:
        refcase = ensemble_config.refcase
        if refcase is None:
            raise ObservationConfigError("REFCASE is required for HISTORY_OBSERVATION")
        error = history_observation.error

        if history_type == HistorySource.REFCASE_HISTORY:
            local_key = history_key(summary_key)
        else:
            local_key = summary_key
        if local_key is None:
            return {}
        if local_key not in refcase.keys:
            return {}
        values = refcase.values[refcase.keys.index(local_key)]
        std_dev = cls._handle_error_mode(values, history_observation)
        for segment_name, segment_instance in history_observation.segment:
            start = segment_instance.start
            stop = segment_instance.stop
            if start < 0:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} out of bounds."
                    " Truncating start of segment to 0.",
                    segment_name,
                )
                start = 0
            if stop >= time_len:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} out of bounds. Truncating"
                    f" end of segment to {time_len - 1}.",
                    segment_name,
                )
                stop = time_len - 1
            if start > stop:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} start after stop. Truncating"
                    f" end of segment to {start}.",
                    segment_name,
                )
                stop = start
            if np.size(std_dev[start:stop]) == 0:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} does not"
                    " contain any time steps. The interval "
                    f"[{start}, {stop}) does not intersect with steps in the"
                    "time map.",
                    segment_name,
                )
            std_dev[start:stop] = cls._handle_error_mode(
                values[start:stop],
                segment_instance,
            )
        data: Dict[Union[int, datetime], Union[GenObservation, SummaryObservation]] = {}
        for i, (date, error, value) in enumerate(zip(refcase.dates, std_dev, values)):
            if error <= std_cutoff:
                ConfigWarning.ert_context_warn(
                    "Too small observation error in observation"
                    f" {summary_key}:{i} - ignored",
                    summary_key,
                )
                continue
            data[date] = SummaryObservation(summary_key, summary_key, value, error)

        return {
            summary_key: ObsVector(
                ResponseTypes.SUMMARY,
                summary_key,
                "summary",
                data,
            )
        }

    @staticmethod
    def _get_time(date_dict: DateValues, start_time: datetime) -> Tuple[datetime, str]:
        if date_dict.date is not None:
            date_str = date_dict.date
            try:
                return datetime.fromisoformat(date_str), f"DATE={date_str}"
            except ValueError:
                try:
                    date = datetime.strptime(date_str, "%d/%m/%Y")
                    ConfigWarning.ert_context_warn(
                        f"Deprecated time format {date_str}."
                        " Please use ISO date format YYYY-MM-DD",
                        date_str,
                    )
                    return date, f"DATE={date_str}"
                except ValueError as err:
                    raise ObservationConfigError.with_context(
                        f"Unsupported date format {date_str}."
                        " Please use ISO date format",
                        date_str,
                    ) from err

        if date_dict.days is not None:
            days = date_dict.days
            return start_time + timedelta(days=days), f"DAYS={days}"
        if date_dict.hours is not None:
            hours = date_dict.hours
            return start_time + timedelta(hours=hours), f"HOURS={hours}"
        raise ValueError("Missing time specifier")

    @staticmethod
    def _find_nearest(
        time_map: List[datetime],
        time: datetime,
        threshold: timedelta = DEFAULT_TIME_DELTA,
    ) -> int:
        nearest_index = -1
        nearest_diff = None
        for i, t in enumerate(time_map):
            diff = abs(time - t)
            if diff < threshold and (nearest_diff is None or nearest_diff > diff):
                nearest_diff = diff
                nearest_index = i
        if nearest_diff is None:
            raise IndexError(f"{time} is not in the time map")
        return nearest_index

    @staticmethod
    def _get_restart(
        date_dict: DateValues,
        obs_name: str,
        time_map: List[datetime],
        has_refcase: bool,
    ) -> int:
        if date_dict.restart is not None:
            return date_dict.restart
        if not time_map:
            raise ObservationConfigError.with_context(
                f"Missing REFCASE or TIME_MAP for observations: {obs_name}",
                obs_name,
            )

        try:
            time, date_str = EnkfObs._get_time(date_dict, time_map[0])
        except ObservationConfigError:
            raise
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Failed to parse date of {obs_name}", obs_name
            ) from err

        try:
            return EnkfObs._find_nearest(time_map, time)
        except IndexError as err:
            raise ObservationConfigError.with_context(
                f"Could not find {time} ({date_str}) in "
                f"the time map for observations {obs_name}"
                + (
                    "The time map is set from the REFCASE keyword. Either "
                    "the REFCASE has an incorrect/missing date, or the observation "
                    "is given an incorrect date.)"
                    if has_refcase
                    else " (The time map is set from the TIME_MAP "
                    "keyword. Either the time map file has an "
                    "incorrect/missing date, or the  observation is given an "
                    "incorrect date."
                ),
                obs_name,
            ) from err

    @staticmethod
    def _make_value_and_std_dev(
        observation_dict: SummaryValues,
    ) -> Tuple[float, float]:
        value = observation_dict.value
        return (
            value,
            float(
                EnkfObs._handle_error_mode(
                    np.array(value),
                    observation_dict,
                )
            ),
        )

    @classmethod
    def _handle_summary_observation(
        cls,
        summary_dict: SummaryValues,
        obs_key: str,
        time_map: List[datetime],
        has_refcase: bool,
    ) -> Dict[str, ObsVector]:
        summary_key = summary_dict.key
        value, std_dev = cls._make_value_and_std_dev(summary_dict)

        try:
            if summary_dict.date is not None and not time_map:
                # We special case when the user has provided date in SUMMARY_OBS
                # and not REFCASE or time_map so that we dont change current behavior.
                try:
                    date = datetime.fromisoformat(summary_dict.date)
                except ValueError as err:
                    raise ValueError("Please use ISO date format YYYY-MM-DD.") from err
                restart = None
            else:
                restart = cls._get_restart(summary_dict, obs_key, time_map, has_refcase)
                date = time_map[restart]
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Problem with date in summary observation {obs_key}: " + str(err),
                obs_key,
            ) from err

        if restart == 0:
            raise ObservationConfigError.with_context(
                "It is unfortunately not possible to use summary "
                "observations from the start of the simulation. "
                f"Problem with observation {obs_key}"
                f"{' at ' + str(cls._get_time(summary_dict, time_map[0])) if summary_dict.restart is None else ''}",
                obs_key,
            )
        return {
            obs_key: ObsVector(
                ResponseTypes.SUMMARY,
                summary_key,
                "summary",
                {date: SummaryObservation(summary_key, obs_key, value, std_dev)},
            )
        }

    @classmethod
    def _create_gen_obs(
        cls,
        scalar_value: Optional[Tuple[float, float]] = None,
        obs_file: Optional[str] = None,
        data_index: Optional[str] = None,
    ) -> GenObservation:
        if scalar_value is None and obs_file is None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if scalar_value is not None and obs_file is not None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if obs_file is not None:
            try:
                file_values = np.loadtxt(obs_file, delimiter=None).ravel()
            except ValueError as err:
                raise ObservationConfigError.with_context(
                    f"Failed to read OBS_FILE {obs_file}: {err}", obs_file
                ) from err
            if len(file_values) % 2 != 0:
                raise ObservationConfigError.with_context(
                    "Expected even number of values in GENERAL_OBSERVATION", obs_file
                )
            values = file_values[::2]
            stds = file_values[1::2]

        else:
            assert scalar_value is not None
            obs_value, obs_std = scalar_value
            values = np.array([obs_value])
            stds = np.array([obs_std])

        if data_index is not None:
            indices = np.array([])
            if os.path.isfile(data_index):
                indices = np.loadtxt(data_index, delimiter=None, dtype=int).ravel()
            else:
                indices = np.array(
                    sorted(rangestring_to_list(data_index)), dtype=np.int32
                )
        else:
            indices = np.arange(len(values))
        std_scaling = np.full(len(values), 1.0)
        if len({len(stds), len(values), len(indices)}) != 1:
            raise ObservationConfigError.with_context(
                f"Values ({values}), error ({stds}) and "
                f"index list ({indices}) must be of equal length",
                obs_file if obs_file is not None else "",
            )
        return GenObservation(values, stds, indices, std_scaling)

    @classmethod
    def _handle_general_observation(
        cls,
        ensemble_config: "EnsembleConfig",
        general_observation: GenObsValues,
        obs_key: str,
        time_map: List[datetime],
        has_refcase: bool,
    ) -> Dict[str, ObsVector]:
        state_kw = general_observation.data
        if not ensemble_config.hasNodeGenData(state_kw):
            ConfigWarning.ert_context_warn(
                f"Ensemble key {state_kw} does not exist"
                f" - ignoring observation {obs_key}",
                state_kw,
            )
            return {}

        if all(
            getattr(general_observation, key) is None
            for key in ["restart", "date", "days", "hours"]
        ):
            # The user has not provided RESTART or DATE, this is legal
            # for GEN_DATA, so we default it to None
            restart = None
        else:
            restart = cls._get_restart(
                general_observation, obs_key, time_map, has_refcase
            )

        config_node = ensemble_config.getNode(state_kw)
        if not isinstance(config_node, GenDataConfig):
            ConfigWarning.ert_context_warn(
                f"{state_kw} has implementation type:"
                f"'{type(config_node)}' - "
                f"expected:'GEN_DATA' in observation:{obs_key}."
                "The observation will be ignored",
                obs_key,
            )
            return {}

        response_report_steps = (
            [] if config_node.report_steps is None else config_node.report_steps
        )
        if (restart is None and response_report_steps) or (
            restart is not None and restart not in response_report_steps
        ):
            ConfigWarning.ert_context_warn(
                f"The GEN_DATA node:{state_kw} is not configured to load from"
                f" report step:{restart} for the observation:{obs_key}"
                " - The observation will be ignored",
                state_kw,
            )
            return {}

        restart = 0 if restart is None else restart
        index_list = general_observation.index_list
        index_file = general_observation.index_file
        if index_list is not None and index_file is not None:
            raise ObservationConfigError.with_context(
                f"GENERAL_OBSERVATION {obs_key} has both INDEX_FILE and INDEX_LIST.",
                obs_key,
            )
        indices = index_list if index_list is not None else index_file
        try:
            return {
                obs_key: ObsVector(
                    ResponseTypes.GEN_DATA,
                    obs_key,
                    config_node.name,
                    {
                        restart: cls._create_gen_obs(
                            (
                                (
                                    general_observation.value,
                                    general_observation.error,
                                )
                                if general_observation.value is not None
                                and general_observation.error is not None
                                else None
                            ),
                            general_observation.obs_file,
                            indices,
                        ),
                    },
                )
            }
        except ValueError as err:
            raise ObservationConfigError.with_context(str(err), obs_key) from err

    def __repr__(self) -> str:
        return f"EnkfObs({self.obs_vectors}, {self.obs_time})"
