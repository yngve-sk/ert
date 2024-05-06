from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union
from uuid import UUID

import numpy
import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from pydantic import BaseModel
from typing_extensions import deprecated

from ert.config.gen_kw_config import GenKwConfig
from ert.config.observations import ObservationsIndices
from ert.storage.mode import BaseMode, Mode, require_write

from ..config import GenDataConfig
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


class ObservationsAndResponsesData:
    def __init__(self, np_arr: npt.NDArray[Any]) -> None:
        self._as_np = np_arr

    def to_long_dataframe(self) -> pd.DataFrame:
        cols = ["key_index", "name", "OBS", "STD", *range(self._as_np.shape[1] - 4)]
        return (
            pd.DataFrame(self._as_np, columns=cols)
            .set_index(["name", "key_index"])
            .astype(float)
        )

    def vec_of_obs_names(self) -> npt.NDArray[np.str_]:
        """
        Extracts a ndarray with the shape (num_obs,).
        Each cell holds the observation name.
        vec_of* getters of this class.
        """
        return self._as_np[:, 1].astype(str)

    def vec_of_errors(self) -> npt.NDArray[np.float_]:
        """
        Extracts a ndarray with the shape (num_obs,).
        Each cell holds the std. error of the observed value.
        The index in this list corresponds to the index of the other
        vec_of* getters of this class.
        """
        return self._as_np[:, 3].astype(float)

    def vec_of_obs_values(self) -> npt.NDArray[np.float_]:
        """
        Extracts a ndarray with the shape (num_obs,).
        Each cell holds the observed value.
        The index in this list corresponds to the index of the other
        vec_of* getters of this class.
        """
        return self._as_np[:, 2].astype(float)

    def vec_of_realization_values(self) -> npt.NDArray[np.float_]:
        """
        Extracts a ndarray with the shape (num_obs, num_reals).
        Each cell holds the response value corresponding to the observation/realization
        indicated by the index. The first index here corresponds to that of other
        vec_of* getters of this class.
        """
        return self._as_np[:, 4:].astype(float)


class LocalEnsemble(BaseMode):
    """
    Represents an ensemble within the local storage system of ERT.

    Manages multiple realizations of experiments, including different sets of
    parameters and responses.
    """

    def __init__(
        self,
        storage: LocalStorage,
        path: Path,
        mode: Mode,
    ):
        """
        Initialize a LocalEnsemble instance.

        Parameters
        ----------
        storage : LocalStorage
            Local storage instance.
        path : Path
            File system path to ensemble data.
        mode : Mode
            Access mode for the ensemble (read/write).
        """

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
        """
        Create a new ensemble in local storage.

        Parameters
        ----------
        storage : LocalStorage
            Local storage instance.
        path : Path
            File system path for ensemble data.
        uuid : UUID
            Unique identifier for the new ensemble.
        ensemble_size : int
            Number of realizations.
        experiment_id : UUID
            Identifier of associated experiment.
        iteration : int
            Iteration number of ensemble.
        name : str
            Name of ensemble.
        prior_ensemble_id : UUID, optional
            Identifier of prior ensemble.

        Returns
        -------
        local_ensemble : LocalEnsemble
            Instance of the newly created ensemble.
        """

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
    def parent(self) -> Optional[UUID]:
        return self._index.prior_ensemble_id

    @property
    def experiment(self) -> LocalExperiment:
        return self._storage.get_experiment(self.experiment_id)

    def get_realization_mask_without_parent_failure(self) -> npt.NDArray[np.bool_]:
        """
        Mask array indicating realizations without a parent failure.

        Returns
        -------
        parent_failures : ndarray of bool
            Boolean array where True means no parent failure.
        """

        return np.array(
            [
                (e != RealizationStorageState.PARENT_FAILURE)
                for e in self.get_ensemble_state()
            ]
        )

    def get_realization_mask_without_failure(self) -> npt.NDArray[np.bool_]:
        """
        Mask array indicating realizations without any failure.

        Returns
        -------
        failures : ndarray of bool
            Boolean array where True means no failure.
        """

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
        """
        Mask array indicating realizations with associated parameters.

        Returns
        -------
        parameters : ndarray of bool
            Boolean array where True means parameters are associated.
        """

        return np.array(
            [
                self._parameters_exist_for_realization(i)
                for i in range(self.ensemble_size)
            ]
        )

    def get_realization_mask_with_responses(
        self, key: Optional[str] = None
    ) -> npt.NDArray[np.bool_]:
        """
        Mask array indicating realizations with associated responses.

        Parameters
        ----------
        key : str, optional
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        masks : ndarray of bool
            Boolean array where True means responses are associated.
        """

        return np.array(
            [
                self._responses_exist_for_realization(i, key)
                for i in range(self.ensemble_size)
            ]
        )

    def _parameters_exist_for_realization(self, realization: int) -> bool:
        """
        Returns true if all parameters in the experiment have
        all been saved in the ensemble. If no parameters, return True

        Parameters
        ----------
        realization : int
            Realization index.

        Returns
        -------
        exists : bool
            True if parameters exist for realization.
        """
        if not self.experiment.parameter_configuration:
            return True
        path = self._realization_dir(realization)
        return all(
            (
                self.has_combined_parameter_dataset(parameter)
                and realization
                in self._load_combined_parameter_dataset(parameter)["realizations"]
            )
            or (path / f"{parameter}.nc").exists()
            for parameter in self.experiment.parameter_configuration
        )

    def has_combined_response_dataset(self, key: str) -> bool:
        ds_key = self._find_unified_dataset_for_response(key)
        return (self._path / f"{ds_key}.nc").exists()

    def has_combined_parameter_dataset(self, key: str) -> bool:
        return (self._path / f"{key}.nc").exists()

    def _load_combined_response_dataset(self, key: str) -> xr.Dataset:
        ds_key = self._find_unified_dataset_for_response(key)

        unified_ds = xr.open_dataset(self._path / f"{ds_key}.nc")

        if key != ds_key:
            return unified_ds.sel(name=key, drop=True)

        return unified_ds

    def _load_combined_parameter_dataset(self, key: str) -> xr.Dataset:
        unified_ds = xr.open_dataset(self._path / f"{key}.nc")

        return unified_ds

    def _responses_exist_for_realization(
        self, realization: int, key: Optional[str] = None
    ) -> bool:
        """
        Returns true if there are responses in the experiment and they have
        all been saved in the ensemble

        Parameters
        ----------
        realization : int
            Realization index.
        key : str, optional
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        exists : bool
            True if responses exist for realization.
        """

        if not self.experiment.response_configuration:
            return True

        real_dir = self._realization_dir(realization)
        if key:
            if self.has_combined_response_dataset(key):
                return (
                    realization
                    in self._load_combined_response_dataset(key)["realization"]
                )
            else:
                return (real_dir / f"{key}.nc").exists()

        return all(
            (real_dir / f"{response}.nc").exists()
            or (
                self.has_combined_response_dataset(response)
                and realization
                in self._load_combined_response_dataset(response)["realization"].values
            )
            for response in self.experiment.response_configuration
        )

    def is_initalized(self) -> List[int]:
        """
        Return the realization numbers where all parameters are internalized. In
        cases where there are parameters which are read from the forward model, an
        ensemble is considered initialized if all other parameters are present

        Returns
        -------
        exists : List[int]
            Returns the realization numbers with parameters
        """

        return list(
            i
            for i in range(self.ensemble_size)
            if all(
                (self._realization_dir(i) / f"{parameter}.nc").exists()
                for parameter in self.experiment.parameter_configuration.values()
                if not parameter.forward_init
            )
            or all(
                (self._path / f"{parameter.name}.nc").exists()
                for parameter in self.experiment.parameter_configuration.values()
                if not parameter.forward_init
            )
        )

    def has_data(self) -> List[int]:
        """
        Return the realization numbers where all responses are internalized

        Returns
        -------
        exists : List[int]
            Returns the realization numbers with responses
        """
        return list(
            i
            for i in range(self.ensemble_size)
            if all(
                self._responses_exist_for_realization(i, response_key)
                for response_key in self.experiment.response_configuration
            )
        )

    def realizations_initialized(self, realizations: List[int]) -> bool:
        """
        Check if specified realizations are initialized.

        Parameters
        ----------
        realizations : list of int
            List of realization indices.

        Returns
        -------
        initialized : bool
            True if all realizations are initialized.
        """

        responses = self.get_realization_mask_with_responses()
        parameters = self.get_realization_mask_with_parameters()

        if len(responses) == 0 and len(parameters) == 0:
            return False

        return all((responses[real] or parameters[real]) for real in realizations)

    def get_realization_list_with_responses(
        self, key: Optional[str] = None
    ) -> List[int]:
        """
        List of realization indices with associated responses.

        Parameters
        ----------
        key : str, optional
            Response key to filter realizations. If None, all responses are considered.

        Returns
        -------
        realizations : list of int
            List of realization indices with associated responses.
        """

        mask = self.get_realization_mask_with_responses(key)
        return np.where(mask)[0].tolist()

    def set_failure(
        self,
        realization: int,
        failure_type: RealizationStorageState,
        message: Optional[str] = None,
    ) -> None:
        """
        Record a failure for a given realization in ensemble.

        Parameters
        ----------
        realization : int
            Index of realization.
        failure_type : RealizationStorageState
            Type of failure.
        message : str, optional
            Optional message describing the failure.
        """

        filename: Path = self._realization_dir(realization) / self._error_log_name
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        error = _Failure(
            type=failure_type, message=message if message else "", time=datetime.now()
        )
        with open(filename, mode="w", encoding="utf-8") as f:
            print(error.model_dump_json(), file=f)

    def unset_failure(
        self,
        realization: int,
    ) -> None:
        filename: Path = self._realization_dir(realization) / self._error_log_name
        if filename.exists():
            filename.unlink()

    def has_failure(self, realization: int) -> bool:
        """
        Check if given realization has a recorded failure.

        Parameters
        ----------
        realization : int
            Index of realization.

        Returns
        -------
        has_failure : bool
            True if realization has a recorded failure.
        """

        return (self._realization_dir(realization) / self._error_log_name).exists()

    def get_failure(self, realization: int) -> Optional[_Failure]:
        """
        Retrieve failure information for a given realization, if any.

        Parameters
        ----------
        realization : int
            Index of realization.

        Returns
        -------
        failure : _Failure, optional
            Failure information if recorded, otherwise None.
        """

        if self.has_failure(realization):
            return _Failure.model_validate_json(
                (self._realization_dir(realization) / self._error_log_name).read_text(
                    encoding="utf-8"
                )
            )
        return None

    def get_ensemble_state(self) -> List[RealizationStorageState]:
        """
        Retrieve the state of each realization within ensemble.

        Returns
        -------
        states : list of RealizationStorageState
            List of realization states.
        """

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

        Returns
        -------
        keys : list of str
            List of summary keys.
        """

        paths_to_check = [*self._path.glob("realization-*/summary.nc")]

        if os.path.exists(self._path / "summary.nc"):
            paths_to_check.append(self._path / "summary.nc")

        for p in paths_to_check:
            return sorted(xr.open_dataset(p)["name"].values)

        return []

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

    def load_parameters(
        self,
        group: str,
        realizations: Union[
            int, npt.NDArray[np.int_], List[int], Tuple[int], None
        ] = None,
    ) -> xr.Dataset:
        """
        Load parameters for group and realizations into xarray Dataset.

        Parameters
        ----------
        group : str
            Name of parameter group to load.
        realizations : {int, ndarray of int}, optional
            Realization indices to load. If None, all realizations are loaded.

        Returns
        -------
        parameters : Dataset
            Loaded xarray Dataset with parameters.
        """

        # UGLY AF, would be better to narrow in realizations
        # to be either None of a tuple of ints
        drop_reals_dim = False
        selected_realizations: Union[None, int, List[int]]
        if realizations is None:
            selected_realizations = None
        elif isinstance(realizations, int):
            assert isinstance(realizations, int)
            drop_reals_dim = True
            selected_realizations = realizations
        elif isinstance(realizations, np.ndarray):
            selected_realizations = realizations.tolist()
        elif isinstance(realizations, tuple):
            selected_realizations = list(realizations)
        elif isinstance(realizations, list):
            selected_realizations = realizations
        else:
            raise ValueError(f"Invalid type for realizations: {type(realizations)}")

        try:
            ds = self.open_unified_parameter_dataset(group)
            if selected_realizations is None:
                return ds

            return ds.sel(realizations=selected_realizations, drop=drop_reals_dim)

        except (ValueError, KeyError, FileNotFoundError):
            # Fallback to check for real folder
            try:
                if selected_realizations is None:
                    return xr.combine_nested(
                        [
                            xr.open_dataset(p)
                            for p in self._path.glob(f"realization-*/{group}.nc")
                        ],
                        concat_dim="realizations",
                    )
                elif isinstance(selected_realizations, int):
                    return xr.open_dataset(
                        self._path
                        / f"realization-{selected_realizations}"
                        / f"{group}.nc"
                    )
                else:
                    assert isinstance(selected_realizations, list)
                    return xr.combine_nested(
                        [
                            xr.open_dataset(
                                self._path / f"realization-{real}" / f"{group}.nc"
                            )
                            for real in selected_realizations
                        ],
                        concat_dim="realizations",
                    )
            except FileNotFoundError as e:
                raise KeyError(
                    f"No dataset '{group}' in storage for "
                    f"realization {selected_realizations}"
                ) from e

    def _find_unified_dataset_for_response(self, key: str) -> str:
        all_gen_data_keys = {
            k
            for k, c in self.experiment.response_configuration.items()
            if isinstance(c, GenDataConfig)
        }

        if key == "gen_data" or key in all_gen_data_keys:
            return "gen_data"

        if key == "summary" or key in self.get_summary_keyset():
            return "summary"

        if key not in self.experiment.response_configuration:
            raise ValueError(f"{key} is not a response")

        return key

    def open_unified_response_dataset(self, key: str) -> xr.Dataset:
        dataset_key = self._find_unified_dataset_for_response(key)
        nc_path = self._path / f"{dataset_key}.nc"

        ds = None
        if os.path.exists(nc_path):
            ds = xr.open_dataset(nc_path)

        if not ds:
            raise FileNotFoundError(
                f"Dataset file for group {key} not found (tried {key}.nc)"
            )

        if key != dataset_key:
            return ds.sel(name=key, drop=True)

        return ds

    def open_unified_parameter_dataset(self, key: str) -> xr.Dataset:
        nc_path = self._path / f"{key}.nc"

        ds = None
        if os.path.exists(nc_path):
            ds = xr.open_dataset(nc_path)

        if not ds:
            raise FileNotFoundError(
                f"Dataset file for group {key} not found (tried {key}.nc)"
            )

        return ds

    def load_responses(
        self, key: str, realizations: Union[Tuple[int], Tuple[int, ...], None] = None
    ) -> xr.Dataset:
        """Load responses for key and realizations into xarray Dataset.

        For each given realization, response data is loaded from the
        file whose filename matches the given key parameter.

        Parameters
        ----------
        key : str
            Response key to load.
        realizations : tuple of int
            Realization indices to load.

        Returns
        -------
        responses : Dataset
            Loaded xarray Dataset with responses.
        """

        try:
            ds = self.open_unified_response_dataset(key)
            if realizations:
                try:
                    return ds.sel(realization=list(realizations))
                except KeyError as err:
                    raise KeyError(
                        f"No response for key {key}, realization: {realizations}"
                    ) from err

            return ds
        except FileNotFoundError:
            # If the unified dataset does not exist,
            # we fall back to checking within the individual realization folders.
            datasets = [
                xr.open_dataset(self._path / f"realization-{real}" / f"{key}.nc")
                for real in (
                    realizations
                    if realizations is not None
                    else self.get_realization_list_with_responses(key)
                )
            ]

            return xr.combine_nested(datasets, concat_dim="realization")

    @deprecated("Use load_responses")
    def load_all_summary_data(
        self,
        keys: Optional[List[str]] = None,
        realization_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load all summary data for realizations into pandas DataFrame.
        Parameters
        ----------
        keys : list of str, optional
            List of keys to load. If None, all keys are loaded.
        realization_index : int, optional
        Returns
        -------
        summary_data : DataFrame
            Loaded pandas DataFrame with summary data.
        """

        try:
            ds = self.load_responses("summary")
            if realization_index is not None:
                ds = ds.sel(realization=realization_index)

            df = ds.to_dataframe().pivot_table(
                index=["realization", "time"], columns="name", values="values"
            )
            df.index = df.index.rename(
                {"time": "Date", "realization": "Realization"}
            ).reorder_levels(["Realization", "Date"])
            df.axes[1].rename("", inplace=True)
            if keys:
                summary_keys = self.get_summary_keyset()
                summary_keys = sorted(
                    [key for key in keys if key in summary_keys]
                )  # ignore keys that doesn't exist
                return df[summary_keys]
            return df
        except (ValueError, FileNotFoundError):
            return pd.DataFrame()

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
        group : str, optional
            Name of parameter group to load.
        relization_index : int, optional
            The realization to load.

        Returns
        -------
        data : DataFrame
            A pandas DataFrame containing the GEN_KW data.

        Notes
        -----
        Any provided keys that are not gen_kw will be ignored.
        """

        dataframes: List[DataFrame] = []
        gen_kws = [
            config
            for config in self.experiment.parameter_configuration.values()
            if isinstance(config, GenKwConfig)
        ]
        if group:
            gen_kws = [config for config in gen_kws if config.name == group]
        for key in gen_kws:
            with contextlib.suppress(KeyError):
                ds = self.load_parameters(key.name)

                if realization_index is not None:
                    ds = ds.sel(realizations=realization_index)

                da = ds["transformed_values"]
                assert isinstance(da, xr.DataArray)
                da["names"] = np.char.add(f"{key.name}:", da["names"].astype(np.str_))
                df = da.to_dataframe().pivot_table(
                    index="realizations", columns="names", values="transformed_values"
                )
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

    def _validate_parameters_dataset(self, group: str, dataset: xr.Dataset) -> None:
        if "values" not in dataset.variables:
            raise ValueError(
                f"Dataset for parameter group '{group}' "
                f"must contain a 'values' variable"
            )

        if dataset["values"].size == 0:
            raise ValueError(
                f"Parameters {group} are empty. Cannot proceed with saving to storage."
            )

        if dataset["values"].ndim >= 2 and dataset["values"].values.dtype == "float64":
            logger.warning(
                "Dataset uses 'float64' for fields/surfaces. Use 'float32' to save memory."
            )

        if group not in self.experiment.parameter_configuration:
            raise ValueError(f"{group} is not registered to the experiment.")

    def _assert_dataset_not_already_created(
        self, group: str, realization: Optional[int] = None
    ) -> None:
        path_in_base_folder = self._path / f"{group}.nc"
        if os.path.exists(path_in_base_folder):
            f"There already exists a combined dataset for parameter group {group}"
            f" for group {group} @ {path_in_base_folder}. Parameters should be saved only once."

        if realization is not None:
            path_in_real_folder = self._realization_dir(realization) / f"{group}.nc"
            if os.path.exists(path_in_real_folder):
                raise KeyError(
                    "Detected attempt at overwriting already saved parameter"
                    f" for group {group} @ {path_in_real_folder}. Parameters should be saved only once."
                )

    def save_parameters_combined(self, group: str, dataset: xr.Dataset) -> None:
        self._validate_parameters_dataset(group, dataset)
        self._assert_dataset_not_already_created(group)
        if "realizations" not in dataset:
            raise KeyError(
                "Combined parameters dataset must have dimension 'realizations'"
            )

        if os.path.exists(self._path / f"{group}.nc"):
            raise KeyError(
                "Detected attempt at saving already saved"
                f"combined dataset for group {group}."
                "Combined datasets should be saved only once."
            )

        dataset.to_netcdf(self._path / f"{group}.nc", engine="scipy")

    @require_write
    def save_parameters(
        self,
        group: str,
        realization: int,
        dataset: xr.Dataset,
    ) -> None:
        """
        Saves the provided dataset under a parameter group and realization index

        Parameters
        ----------
        group : str
            Parameter group name for saving dataset.

        realization : int
            Realization index for saving group.

        dataset : Dataset
            Dataset to save. It must contain a variable named 'values'
            which will be used when flattening out the parameters into
            a 1d-vector.
        """

        self._assert_dataset_not_already_created(group)
        self._validate_parameters_dataset(group, dataset)

        path = self._realization_dir(realization) / f"{group}.nc"
        path.parent.mkdir(exist_ok=True)

        if "realizations" not in dataset.dims:
            dataset = dataset.expand_dims(realizations=[realization])

        dataset.to_netcdf(path, engine="scipy")

    @require_write
    def save_response(self, group: str, data: xr.Dataset, realization: int) -> None:
        """
        Save dataset as response under group and realization index.

        Parameters
        ----------
        group : str
            Response group name for saving dataset.
        realization : int
            Realization index for saving group.
        data : Dataset
            Dataset to save.
        """

        if "values" not in data.variables:
            raise ValueError(
                f"Dataset for response group '{group}' "
                f"must contain a 'values' variable"
            )

        if data["values"].size == 0:
            raise ValueError(
                f"Responses {group} are empty. Cannot proceed with saving to storage."
            )

        if "realization" not in data.dims:
            data = data.expand_dims({"realization": [realization]})

        output_path = self._realization_dir(realization)
        Path.mkdir(output_path, parents=True, exist_ok=True)

        data.to_netcdf(output_path / f"{group}.nc", engine="scipy")

    def calculate_std_dev_for_parameter(self, parameter_group: str) -> xr.Dataset:
        if not parameter_group in self.experiment.parameter_configuration:
            raise ValueError(f"{parameter_group} is not registered to the experiment.")

        path_unified = self._path / f"{parameter_group}.nc"
        if os.path.exists(path_unified):
            return xr.open_dataset(path_unified).std("realizations")

        path = self._path / "realization-*" / f"{parameter_group}.nc"
        try:
            ds = xr.open_mfdataset(str(path))
        except OSError as e:
            raise e

        return ds.std("realizations")

    def get_measured_data(
        self,
        observation_keys: List[str],
        active_realizations: Optional[npt.NDArray[np.int_]] = None,
    ) -> ObservationsAndResponsesData:
        """Return data grouped by observation name, showing the
        observation + std values, and accompanying simulated values per realization.

        * key_index is the "{time}" for summary, "{index},{report_step}" for gen_obs
        * Numbers 0...N correspond to the realization index

        Example output
                                  OBS  STD          0  ...        48         49
        name     key_index                             ...
        POLY_OBS [0, 0]      2.145705  0.6   3.721637  ...  0.862469   2.625992
                 [2, 0]      8.769220  1.4   6.419814  ...  1.304883   4.650068
                 [4, 0]     12.388015  3.0  12.796416  ...  2.535165   8.349348
                 [6, 0]     25.600465  5.4  22.851445  ...  4.553314  13.723831
                 [8, 0]     42.352048  8.6  36.584901  ...  7.359332  20.773518

        Arguments:
            observation_keys: List of observation names to include in the dataset
            active_realizations: List of active realization indices
        """

        long_nps = []
        reals_with_responses_mask = self.get_realization_list_with_responses()
        if active_realizations is not None:
            reals_with_responses_mask = np.intersect1d(
                active_realizations, np.array(reals_with_responses_mask)
            )

        # Ensure to sort keys at all levels to preserve deterministic ordering
        # Traversal will be in this order:
        # response_type -> obs name -> response name
        for response_type in sorted(self.experiment.observations):
            obs_datasets = self.experiment.observations[response_type]
            # obs_keys_ds = xr.Dataset({"obs_name": observation_keys})
            obs_names_to_check = set(obs_datasets["obs_name"].data).intersection(
                observation_keys
            )
            responses_ds = self.load_responses(
                response_type,
                realizations=tuple(reals_with_responses_mask),
            )

            index = ObservationsIndices[response_type]
            for obs_name in sorted(obs_names_to_check):
                obs_ds = obs_datasets.sel(obs_name=obs_name, drop=True)

                obs_ds = obs_ds.dropna("name", subset=["observations"], how="all")
                for k in index:
                    obs_ds = obs_ds.dropna(dim=k, how="all")

                response_names_to_check = obs_ds["name"].data

                for response_name in sorted(response_names_to_check):
                    observations_for_response = obs_ds.sel(
                        name=response_name, drop=True
                    )

                    responses_matching_obs = responses_ds.sel(
                        name=response_name, drop=True
                    )

                    combined = observations_for_response.merge(
                        responses_matching_obs, join="left"
                    )

                    response_vals_per_real = (
                        combined["values"].stack(key=index).values.T
                    )

                    key_index_1d = np.array(
                        [
                            (
                                x.strftime("%Y-%m-%d")
                                if isinstance(x, pd.Timestamp)
                                else json.dumps(x)
                            )
                            for x in combined[index].coords.to_index()
                        ]
                    ).reshape(-1, 1)
                    obs_vals_1d = combined["observations"].data.reshape(-1, 1)
                    std_vals_1d = combined["std"].data.reshape(-1, 1)

                    num_obs_names = len(obs_vals_1d)
                    obs_names_1d = np.full((len(std_vals_1d), 1), obs_name)

                    if (
                        len(key_index_1d) != num_obs_names
                        or len(response_vals_per_real) != num_obs_names
                        or len(obs_names_1d) != num_obs_names
                        or len(std_vals_1d) != num_obs_names
                    ):
                        raise IndexError(
                            "Axis 0 misalignment, expected axis0 length to "
                            f"correspond to observation names {num_obs_names}. Got:\n"
                            f"len(response_vals_per_real)={len(response_vals_per_real)}\n"
                            f"len(obs_names_1d)={len(obs_names_1d)}\n"
                            f"len(std_vals_1d)={len(std_vals_1d)}"
                        )

                    if response_vals_per_real.shape[1] != len(
                        reals_with_responses_mask
                    ):
                        raise IndexError(
                            "Axis 1 misalignment, expected axis 1 of"
                            f" response_vals_per_real to be the same as number of realizations"
                            f" with responses ({len(reals_with_responses_mask)}),"
                            f"but got response_vals_per_real.shape[1]"
                            f"={response_vals_per_real.shape[1]}"
                        )

                    combined_np_long = np.concatenate(
                        [
                            key_index_1d,
                            obs_names_1d,
                            obs_vals_1d,
                            std_vals_1d,
                            response_vals_per_real,
                        ],
                        axis=1,
                    )
                    long_nps.append(combined_np_long)

        if not long_nps:
            msg = (
                "No observation: "
                + (", ".join(observation_keys) if observation_keys is not None else "*")
                + " in ensemble"
            )
            raise KeyError(msg)

        long_np = numpy.concatenate(long_nps)

        return ObservationsAndResponsesData(long_np)

    @staticmethod
    def _ensure_correct_coordinate_order(ds: xr.Dataset) -> xr.Dataset:
        """
        Ensures correct coordinate order or response/param dataset.
        Slightly less performant than not doing it, but ensure the
        correct coordinate order is applied when doing .to_dataframe().
        It is possible to omit using this and instead pass in the correct
        dim order when doing .to_dataframe(), which is always the same as
        the .dims of the first data var of this dataset.
        """
        # Just to make the order right when
        # doing .to_dataframe()
        # (it seems notoriously hard to tell xarray to just reorder
        # the dimensions/coordinate labels)
        data_vars = list(ds.data_vars.keys())

        # We assume only data vars with the same dimensions,
        # i.e., (realization, *index) for all of them.
        dim_order_of_first_var = ds[data_vars[0]].dims
        return ds[[*dim_order_of_first_var, *data_vars]].sortby(
            dim_order_of_first_var[0]  # "realization" / "realizations"
        )

    def _unify_datasets(
        self,
        groups: List[str],
        concat_dim: Literal["realization", "realizations"],
        delete_after: bool = True,
    ) -> None:
        for group in groups:
            combined_ds_path = self._path / f"{group}.nc"
            has_existing_combined = os.path.exists(combined_ds_path)

            paths = sorted(self.mount_point.glob(f"realization-*/{group}.nc"))

            if len(paths) > 0:
                new_combined = xr.combine_nested(
                    [xr.open_dataset(p, engine="scipy") for p in paths],
                    concat_dim=concat_dim,
                )

                if has_existing_combined:
                    # Merge new combined into old
                    old_combined = xr.open_dataset(combined_ds_path)
                    reals_to_replace = new_combined[concat_dim].data
                    new_combined = old_combined.drop_sel(
                        {concat_dim: reals_to_replace}
                    ).merge(new_combined)
                    os.remove(combined_ds_path)

                new_combined = self._ensure_correct_coordinate_order(new_combined)

                if not new_combined:
                    raise ValueError("Unified dataset somehow ended up empty")

                new_combined.to_netcdf(combined_ds_path, engine="scipy")

                if delete_after:
                    for p in paths:
                        os.remove(p)

    def unify_responses(self, key: Optional[str] = None) -> None:
        if key is None:
            for key in self.experiment.response_configuration:
                self.unify_responses(key)

        gen_data_keys = {
            k
            for k, c in self.experiment.response_configuration.items()
            if isinstance(c, GenDataConfig)
        }

        if key == "gen_data" or key in gen_data_keys:
            has_existing_combined = os.path.exists(self._path / "gen_data.nc")

            # If gen data, combine across reals,
            # but also across all name(s) into one gen_data.nc

            files_to_remove = []
            to_concat = []
            for group in gen_data_keys:
                paths = sorted(self.mount_point.glob(f"realization-*/{group}.nc"))

                if len(paths) > 0:
                    ds_for_group = xr.concat(
                        [
                            ds.expand_dims(name=[group], axis=1)
                            for ds in [
                                xr.open_dataset(p, engine="scipy") for p in paths
                            ]
                        ],
                        dim="realization",
                    )
                    to_concat.append(ds_for_group)

                    files_to_remove.extend(paths)

            # Ensure deterministic ordering wrt name and real
            if to_concat:
                new_combined_ds = xr.concat(to_concat, dim="name").sortby(
                    ["realization", "name"]
                )
                new_combined_ds = self._ensure_correct_coordinate_order(new_combined_ds)

                if has_existing_combined:
                    old_combined = xr.load_dataset(self._path / "gen_data.nc")
                    updated_realizations = new_combined_ds["realization"].data
                    new_combined_ds = old_combined.drop_sel(
                        {"realization": updated_realizations}
                    ).merge(new_combined_ds)
                    os.remove(self._path / "gen_data.nc")

                new_combined_ds.to_netcdf(self._path / "gen_data.nc", engine="scipy")
                for f in files_to_remove:
                    os.remove(f)

        else:
            # If it is a summary, just combined across reals
            self._unify_datasets(
                (
                    [key]
                    if key is not None
                    else list(self.experiment.response_configuration.keys())
                ),
                "realization",
            )

    def unify_parameters(self, key: Optional[str] = None) -> None:
        self._unify_datasets(
            (
                [key]
                if key is not None
                else list(self.experiment.parameter_configuration.keys())
            ),
            "realizations",
        )
