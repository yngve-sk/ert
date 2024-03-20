from __future__ import annotations

import contextlib
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from typing_extensions import deprecated

from ert.config.gen_data_config import GenDataConfig
from ert.config.gen_kw_config import GenKwConfig
from ert.storage.mode import BaseMode, Mode, require_write

from .realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage.local_experiment import LocalExperiment
    from ert.storage.local_storage import LocalStorage

logger = logging.getLogger(__name__)


class _Index(BaseModel):
    id: UUID
    experiment_id: UUID
    ensemble_size: int
    iteration: int
    name: str
    prior_ensemble_id: Optional[UUID]
    started_at: datetime


class _Failure(BaseModel):
    type: RealizationStorageState
    message: str
    time: datetime


class LocalEnsemble(BaseMode):
    def __init__(
        self,
        storage: LocalStorage,
        path: Path,
        mode: Mode,
    ):
        super().__init__(mode)
        self._storage = storage
        self._path = path

        self._index = _Index.model_validate_json(
            (path / "index.json").read_text(encoding="utf-8")
        )
        self._error_log_name = "error.json"

        @lru_cache(maxsize=None)
        def create_realization_dir(realization: int) -> Path:
            return self._path / f"realization-{realization}"

        self._realization_dir = create_realization_dir

    @classmethod
    def create(
        cls,
        storage: LocalStorage,
        path: Path,
        uuid: UUID,
        *,
        ensemble_size: int,
        experiment_id: UUID,
        iteration: int = 0,
        name: str,
        prior_ensemble_id: Optional[UUID],
    ) -> LocalEnsemble:
        (path / "experiment").mkdir(parents=True, exist_ok=False)

        index = _Index(
            id=uuid,
            ensemble_size=ensemble_size,
            experiment_id=experiment_id,
            iteration=iteration,
            name=name,
            prior_ensemble_id=prior_ensemble_id,
            started_at=datetime.now(),
        )

        with open(path / "index.json", mode="w", encoding="utf-8") as f:
            print(index.model_dump_json(), file=f)

        return cls(storage, path, Mode.WRITE)

    @property
    def mount_point(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._index.name

    @property
    def id(self) -> UUID:
        return self._index.id

    @property
    def experiment_id(self) -> UUID:
        return self._index.experiment_id

    @property
    def ensemble_size(self) -> int:
        return self._index.ensemble_size

    @property
    def started_at(self) -> datetime:
        return self._index.started_at

    @property
    def iteration(self) -> int:
        return self._index.iteration

    @property
    def experiment(self) -> LocalExperiment:
        return self._storage.get_experiment(self.experiment_id)

    def get_realization_mask_without_parent_failure(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [
                (e != RealizationStorageState.PARENT_FAILURE)
                for e in self.get_ensemble_state()
            ]
        )

    def get_realization_mask_without_failure(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [
                e
                not in [
                    RealizationStorageState.PARENT_FAILURE,
                    RealizationStorageState.LOAD_FAILURE,
                ]
                for e in self.get_ensemble_state()
            ]
        )

    def get_realization_mask_with_parameters(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [
                self._parameters_exist_for_realization(i)
                for i in range(self.ensemble_size)
            ]
        )

    def get_realization_mask_with_responses(
        self, key: Optional[str] = None
    ) -> npt.NDArray[np.bool_]:
        return np.array(
            [
                self._responses_exist_for_realization(i, key)
                for i in range(self.ensemble_size)
            ]
        )

    def _parameters_exist_for_realization(self, realization: int) -> bool:
        """
        Returns true if there are parameters in the experiment and they have
        all been saved in the ensemble
        """
        if not self.experiment.parameter_configuration:
            return False

        path = self._realization_dir(realization)
        return all(
            (
                self._has_combined_dataset(parameter)
                and realization in self.load_combined_dataset(parameter)["realizations"]
            )
            or (path / f"{parameter}.nc").exists()
            for parameter in self.experiment.parameter_configuration
        )

    def _has_combined_dataset(self, key: str) -> bool:
        return (self._path / f"{key}.nc").exists() or (
            self._path / f"{key}.nc"
        ).exists()

    @lru_cache  # noqa: B019
    def load_combined_dataset(self, key: str) -> xr.Dataset:
        return xr.open_dataset(self._path / f"{key}.nc")

    def _responses_exist_for_realization(
        self, realization: int, key: Optional[str] = None
    ) -> bool:
        """
        Returns true if there are responses in the experiment and they have
        all been saved in the ensemble
        """
        if not self.experiment.response_configuration:
            return False

        real_dir = self._realization_dir(realization)
        if key:
            if self._has_combined_dataset(key):
                return realization in self.load_combined_dataset(key)["realization"]
            else:
                return (real_dir / f"{key}.nc").exists()

        return all(
            (real_dir / f"{response}.nc").exists()
            or (
                self._has_combined_dataset(response)
                and realization
                in self.load_combined_dataset(response)["realization"].values
            )
            for response in self.experiment.response_configuration
        )

    def is_initalized(self) -> bool:
        """
        Check that the ensemble has all parameters present in at least one realization
        """
        return any(
            all(
                (self._realization_dir(i) / f"{parameter}.nc").exists()
                for parameter in self.experiment.parameter_configuration
            )
            for i in range(self.ensemble_size)
        )

    def has_data(self) -> bool:
        """
        Check that the ensemble has all responses present in at least one realization
        """
        return any(
            (
                all(
                    (self._realization_dir(i) / f"{response}.nc").exists()
                    for response in self.experiment.response_configuration
                )
                or all(
                    self._has_combined_dataset(response)
                    for response in self.experiment.response_configuration
                )
            )
            for i in range(self.ensemble_size)
        )

    def realizations_initialized(self, realizations: List[int]) -> bool:
        responses = self.get_realization_mask_with_responses()
        parameters = self.get_realization_mask_with_parameters()

        if len(responses) == 0 and len(parameters) == 0:
            return False

        return all((responses[real] or parameters[real]) for real in realizations)

    def get_realization_list_with_responses(
        self, key: Optional[str] = None
    ) -> List[int]:
        mask = self.get_realization_mask_with_responses(key)
        return np.where(mask)[0].tolist()

    def set_failure(
        self,
        realization: int,
        failure_type: RealizationStorageState,
        message: Optional[str] = None,
    ) -> None:
        filename: Path = self._realization_dir(realization) / self._error_log_name
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        error = _Failure(
            type=failure_type, message=message if message else "", time=datetime.now()
        )
        with open(filename, mode="w", encoding="utf-8") as f:
            print(error.model_dump_json(), file=f)

    def has_failure(self, realization: int) -> bool:
        return (self._realization_dir(realization) / self._error_log_name).exists()

    def get_failure(self, realization: int) -> Optional[_Failure]:
        if self.has_failure(realization):
            return _Failure.model_validate_json(
                (self._realization_dir(realization) / self._error_log_name).read_text(
                    encoding="utf-8"
                )
            )
        return None

    def get_ensemble_state(self) -> List[RealizationStorageState]:
        def _find_state(realization: int) -> RealizationStorageState:
            if self.has_failure(realization):
                failure = self.get_failure(realization)
                assert failure
                return failure.type
            if self._responses_exist_for_realization(realization):
                return RealizationStorageState.HAS_DATA
            if self._parameters_exist_for_realization(realization):
                return RealizationStorageState.INITIALIZED
            else:
                return RealizationStorageState.UNDEFINED

        return [_find_state(i) for i in range(self.ensemble_size)]

    def get_summary_keyset(self) -> List[str]:
        """
        Find the first folder with summary data then load the
        summary keys from this.
        """

        try:
            summary_data = self.load_responses(
                "summary",
                tuple(self.get_realization_list_with_responses("summary")),
            )
            return sorted(summary_data["name"].values)
        except (ValueError, KeyError):
            return []

    def _get_gen_data_config(self, key: str) -> GenDataConfig:
        config = self.experiment.response_configuration[key]
        assert isinstance(config, GenDataConfig)
        return config

    def _load_single_dataset(
        self,
        group: str,
        realization: int,
    ) -> xr.Dataset:
        try:
            return xr.open_dataset(
                self.mount_point / f"realization-{realization}" / f"{group}.nc",
                engine="scipy",
            )
        except FileNotFoundError as e:
            raise KeyError(
                f"No dataset '{group}' in storage for realization {realization}"
            ) from e

    def _ensure_unified_dataset_exists(self, group: str):
        try:
            self.open_unified_dataset(group)
        except FileNotFoundError:
            if group in self.experiment.response_info:
                self._unify_parameters(group)
            else:
                self._unify_responses(group)

    def load_parameters(
        self,
        group: str,
        realizations: Union[int, Tuple[int], npt.NDArray[np.int_], None] = None,
    ) -> xr.Dataset:
        return self._load_dataset(group, realizations)

        try:
            self.open_unified_dataset(group)
        except FileNotFoundError:
            self._unify_parameters()
        self._ensure_unified_dataset_exists(group)

        try:
            ds = self.open_unified_dataset(group)

            if realizations is not None:
                return ds.sel(
                    realizations=(
                        list(realizations)
                        if type(realizations) is not int
                        else realizations
                    )
                )
            return ds
        except (KeyError, FileNotFoundError) as e:
            raise KeyError(
                f"No dataset '{group}' in storage for realization {realizations}"
            ) from e

    def open_unified_dataset(self, key: str) -> xr.Dataset:
        nc_path = self._path / f"{key}.nc"

        if os.path.exists(nc_path):
            return xr.open_dataset(nc_path)

        raise FileNotFoundError(
            f"Dataset file for group {key} not found (tried {key}.nc)"
        )

    @lru_cache  # noqa: B019
    def load_responses(self, key: str, realizations: Tuple[int]) -> xr.Dataset:
        if key not in self.experiment.response_info:
            raise ValueError(f"{key} is not a response")
        self._ensure_unified_dataset_exists(key)

        ds = self.open_unified_dataset(key)
        if realizations:
            try:
                return ds.sel(realization=list(realizations))
            except KeyError as err:
                raise KeyError(
                    f"No response for key {key}, realization: {realizations}"
                ) from err

        return ds

    @deprecated("Use load_responses")
    def load_all_summary_data(
        self,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        realizations = self.get_realization_list_with_responses()
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]

        summary_keys = self.get_summary_keyset()

        try:
            df = self.load_responses("summary", tuple(realizations)).to_dataframe()
        except (ValueError, KeyError):
            return pd.DataFrame()

        df = df.unstack(level="name")
        df.columns = [col[1] for col in df.columns.values]
        df.index = df.index.rename(
            {"time": "Date", "realization": "Realization"}
        ).reorder_levels(["Realization", "Date"])
        if keys:
            summary_keys = sorted(
                [key for key in keys if key in summary_keys]
            )  # ignore keys that doesn't exist
            return df[summary_keys]
        return df

    def load_all_gen_kw_data(
        self,
        group: Optional[str] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads scalar parameters (GEN_KWs) into a pandas DataFrame
        with columns <PARAMETER_GROUP>:<PARAMETER_NAME> and
        "Realization" as index.

        Parameters
        ----------
        group: str, optional
            Name of parameter group to load.
        relization_index: int, optional
            The realization to load.

        Returns
        -------
        DataFrame:
            A pandas DataFrame containing the GEN_KW data.

        Note
        ----
            Any provided keys that are not gen_kw will be ignored.
        """
        if realization_index is not None:
            realizations = np.array([realization_index])
        else:
            ens_mask = (
                self.get_realization_mask_with_responses()
                + self.get_realization_mask_with_parameters()
            )
            realizations = np.flatnonzero(ens_mask)

        dataframes = []
        gen_kws = [
            config
            for config in self.experiment.parameter_configuration.values()
            if isinstance(config, GenKwConfig)
        ]
        if group:
            gen_kws = [config for config in gen_kws if config.name == group]
        for key in gen_kws:
            with contextlib.suppress(KeyError):
                da = self.load_parameters(key.name, realizations)["transformed_values"]
                assert isinstance(da, xr.DataArray)
                da["names"] = np.char.add(f"{key.name}:", da["names"].astype(np.str_))
                df = da.to_dataframe().unstack(level="names")
                df.columns = df.columns.droplevel()
                for parameter in df.columns:
                    if key.shouldUseLogScale(parameter.split(":")[1]):
                        df[f"LOG10_{parameter}"] = np.log10(df[parameter])
                dataframes.append(df)
        if not dataframes:
            return pd.DataFrame()

        dataframe = pd.concat(dataframes, axis=1)
        dataframe.columns.name = None
        dataframe.index.name = "Realization"

        return dataframe.sort_index(axis=1)

    @require_write
    def save_parameters(
        self,
        group: str,
        realization: int,
        dataset: xr.Dataset,
    ) -> None:
        """Saves the provided dataset under a parameter group and realization index

        Args:
            group: Name of the parameter group under which the dataset is to be saved

            realization: Which realization index this group belongs to

            dataset: Dataset to save. It must contain a variable named
                    'values' which will be used when flattening out the
                    parameters into a 1d-vector.
        """
        if "values" not in dataset.variables:
            raise ValueError(
                f"Dataset for parameter group '{group}' "
                f"must contain a 'values' variable"
            )

        if dataset["values"].ndim >= 2 and dataset["values"].values.dtype == "float64":
            logger.warning(
                "Dataset uses 'float64' for fields/surfaces. Use 'float32' to save memory."
            )

        if group not in self.experiment.parameter_configuration:
            raise ValueError(f"{group} is not registered to the experiment.")

        path = self._realization_dir(realization) / f"{group}.nc"
        path.parent.mkdir(exist_ok=True)

        dataset.expand_dims(realizations=[realization]).to_netcdf(path, engine="scipy")

        if os.path.exists(self._path / f"{group}.nc"):
            os.remove(self._path / f"{group}.nc")

    @require_write
    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        if "realization" not in data.dims:
            data = data.expand_dims({"realization": [realization]})

        output_path = self._realization_dir(realization)
        Path.mkdir(output_path, parents=True, exist_ok=True)

        data.to_netcdf(output_path / f"{group}.nc", engine="scipy")

    def calculate_std_dev_for_parameter(self, parameter_group: str) -> xr.Dataset:
        if not parameter_group in self.experiment.parameter_configuration:
            raise ValueError(f"{parameter_group} is not registered to the experiment.")

        path = self._path / "realization-*" / f"{parameter_group}.nc"
        try:
            ds = xr.open_mfdataset(str(path))
        except OSError as e:
            raise e

        return ds.std("realizations")

        if os.path.exists(self._path / f"{group}.nc"):
            # Ideally this should never happen
            os.remove(self._path / f"{group}.nc")

    def _unify_datasets(
        self,
        groups: List[str],
        concat_dim: Literal["realization", "realizations"],
        delete_after: bool = True,
    ) -> None:
        for group in groups:
            paths = sorted(self.mount_point.glob(f"realization-*/{group}.nc"))

            if len(paths) > 0:
                xr.combine_nested(
                    [xr.open_dataset(p, engine="scipy") for p in paths],
                    concat_dim=concat_dim,
                ).to_netcdf(self._path / f"{group}.nc", engine="scipy")

                if delete_after:
                    for p in paths:
                        os.remove(p)

    def _unify_responses(self, key: Optional[str] = None) -> None:
        self._unify_datasets(
            [key] if key is not None else list(self.experiment.response_info.keys()),
            "realization",
        )

    def _unify_parameters(self, key: Optional[str] = None) -> None:
        self._unify_datasets(
            [key] if key is not None else list(self.experiment.parameter_info.keys()),
            "realizations",
        )
