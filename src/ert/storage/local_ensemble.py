from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime
from functools import lru_cache, reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import UUID

import numpy
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from typing_extensions import deprecated

from ert.config.gen_kw_config import GenKwConfig
from ert.config.responses.observations import ObservationsIndices
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


class InMemoryStorageForSmallDatasets:
    """
    Store minimally sized datasets in-memory to avoid unnecessary read+write
    """

    def __init__(
        self,
        max_bytes_total: int,
        max_bytes_per_ds: int,
        base_dir: Optional[Path] = None,
    ):
        self._max_bytes_per_ds = max_bytes_per_ds
        self._max_bytes_total = max_bytes_total
        self.nbytes = 0
        self._datasets: Dict[str, List[Tuple[int, xr.Dataset]]] = {}
        self._datasets_in_tmpfile: Dict[str, List[Tuple[List[int], xr.Dataset]]] = {}

        if base_dir is not None:
            self._tmpdir = os.path.join(base_dir, "small_datasets_tmp_storage")

            if not os.path.exists(self._tmpdir):
                os.mkdir(self._tmpdir)
        else:
            # Indicates no use of temp folder to dump accumulated small datasets
            self._tmpdir = None

        self._file_counter = 0  # Only needed for unique filenames

    def store_to_tmpfiles(self):
        if self._tmpdir is not None:
            groups_to_remove = []
            for group, real_ds_tuples in self._datasets.items():
                datasets = [ds for _, ds in real_ds_tuples]
                first_real, first_ds = real_ds_tuples[0]
                all_reals = [real for real, _ in real_ds_tuples]

                file_no = self._file_counter
                self._file_counter += 1
                filepath = os.path.join(
                    self._tmpdir,
                    f"chunk_{file_no}_{group}.nc",
                )
                # Can remove this piece if we name the realizations
                # dimension the same in params and responses.
                real_key = (
                    "realization" if "realization" in first_ds else "realizations"
                )
                combined_ds = xr.combine_nested(datasets, concat_dim=real_key)
                combined_ds.to_netcdf(filepath, engine="scipy")

                if group not in self._datasets_in_tmpfile:
                    self._datasets_in_tmpfile[group] = []

                ds_handle_to_file = xr.open_dataset(filepath)

                # Should not be necessary due to xr lazy load
                # but doing this just to be sure...
                ds_handle_to_file.close()
                self._datasets_in_tmpfile[group].append((all_reals, ds_handle_to_file))
                self.nbytes -= reduce(lambda p, ds: p + ds.nbytes, datasets, 0)
                groups_to_remove.append(group)

            for group in groups_to_remove:
                del self._datasets[group]

            assert self.nbytes == 0

    def try_store_in_memory(self, group: str, realization: int, ds: xr.Dataset) -> bool:
        if ds.nbytes > self._max_bytes_per_ds:
            return False

        if self.nbytes + ds.nbytes > self._max_bytes_total:
            if self._tmpdir is not None:
                self.store_to_tmpfiles()
            else:
                return False

        self.nbytes += ds.nbytes

        if group not in self._datasets:
            self._datasets[group] = []

        self._datasets[group].append((realization, ds))

        return True

    def has(self, group: str, realization: int) -> bool:
        return (
            group in self._datasets
            and any(r == realization for r, _ in self._datasets[group])
        ) or (
            group in self._datasets_in_tmpfile
            and any(
                realization in reals for reals, _ in self._datasets_in_tmpfile[group]
            )
        )

    def consume_group_if_exists(self, group: str) -> List[xr.Dataset]:
        datasets = self._datasets.get(group, [])

        group_size = reduce(lambda p, ds: p + ds.nbytes, [ds for _, ds in datasets], 0)
        self.nbytes -= group_size

        from_memory = [ds for _, ds in datasets]
        from_files = []

        if group in self._datasets_in_tmpfile:
            for _, ds_handle in self._datasets_in_tmpfile[group]:
                # Assumption: It is always OK to load the datasets
                # into memory for one group, as one group can't possibly
                # be too large to contain in memory.
                # Datasets stored here are always below a certain
                # max size limit anyway.
                # IF it is too large, we would have to destroy the file after
                # putting it into the combined one w/ lazy load
                from_files.append(ds_handle.load())

            del self._datasets_in_tmpfile[group]

        return [*from_memory, *from_files]


class LocalEnsemble(BaseMode):
    IN_MEMORY_STORAGE_MAX_SIZE = int(1e9)
    IN_MEMORY_STORAGE_MAX_SIZE_PER_DATASET = int(1e6)
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
        self._in_memory_datasets = InMemoryStorageForSmallDatasets(
            max_bytes_total=LocalEnsemble.IN_MEMORY_STORAGE_MAX_SIZE,
            max_bytes_per_ds=LocalEnsemble.IN_MEMORY_STORAGE_MAX_SIZE_PER_DATASET,
            base_dir=self._path,
        )

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
        Returns true if there are parameters in the experiment and they have
        all been saved in the ensemble

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
            return False

        path = self._realization_dir(realization)
        return all(
            (
                self._has_combined_parameter_dataset(parameter)
                and realization
                in self._load_combined_parameter_dataset(parameter)["realizations"]
            )
            or (path / f"{parameter}.nc").exists()
            or self._in_memory_datasets.has(parameter, realization)
            for parameter in self.experiment.parameter_configuration
        )

    def _has_combined_response_dataset(self, key: str) -> bool:
        ds_key = self._find_unified_dataset_for_response(key)
        return (self._path / f"{ds_key}.nc").exists()

    def _has_combined_parameter_dataset(self, key: str) -> bool:
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
            return False

        real_dir = self._realization_dir(realization)
        if key:
            if self._has_combined_response_dataset(key):
                return (
                    realization
                    in self._load_combined_response_dataset(key)["realization"]
                )
            else:
                return (
                    real_dir / f"{key}.nc"
                ).exists() or self._in_memory_datasets.has(key, realization)

        return all(
            self._in_memory_datasets.has(response, realization)
            or (real_dir / f"{response}.nc").exists()
            or (
                self._has_combined_response_dataset(response)
                and realization
                in self._load_combined_response_dataset(response)["realization"].values
            )
            for response in self.experiment.response_configuration
        )

    def is_initalized(self) -> bool:
        """
        Check that the ensemble has all parameters present in at least one realization

        Returns
        -------
        exists : bool
            True if all parameters are present in at least one realization.
        """

        return any(
            (
                all(
                    (self._realization_dir(i) / f"{param}.nc").exists()
                    for param in self.experiment.parameter_configuration
                )
            )
            for i in range(self.ensemble_size)
        ) or all(
            (self._path / f"{param}.nc").exists()
            for param in self.experiment.parameter_configuration
        )

    def _has_response_for_at_least_one_realization(self, response_key: str) -> bool:
        ds_key = self._find_unified_dataset_for_response(response_key)

        if (self._path / f"{ds_key}.nc").exists():
            # We assume it exists in the unified ds
            return True

        return any(
            (self._realization_dir(i) / f"{response_key}.nc").exists()
            or self._in_memory_datasets.has(response_key, i)
            for i in range(self.ensemble_size)
        )

    def has_data(self) -> bool:
        """
        Check that the ensemble has all responses present in at least one realization

        Returns
        -------
        exists : bool
            True if all responses are present in at least one realization.
        """
        return all(
            self._has_response_for_at_least_one_realization(response_key)
            for response_key in self.experiment.response_configuration
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

    def _ensure_unified_parameter_dataset_exists(self, group: str) -> None:
        try:
            self.open_unified_parameter_dataset(group)
        except FileNotFoundError:
            self.unify_parameters(group)

    def _ensure_unified_response_dataset_exists(self, group: str) -> None:
        try:
            self.open_unified_response_dataset(group)
        except FileNotFoundError:
            self.unify_responses(group)

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

        self._ensure_unified_parameter_dataset_exists(group)

        try:
            ds = self.open_unified_parameter_dataset(group)

            if realizations is not None:
                realizations_list = realizations
                if type(realizations) is int:
                    assert type(realizations) is int
                    return ds.sel(realizations=realizations)
                elif type(realizations) is np.ndarray:
                    realizations_list = realizations.tolist()
                elif isinstance(realizations, tuple):
                    realizations_list = list(realizations)

                return ds.sel(realizations=realizations_list)

            return ds
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise KeyError(
                f"No dataset '{group}' in storage for realization {realizations}"
            ) from e

    def _find_unified_dataset_for_response(self, key: str) -> str:
        all_gen_data_keys = {
            k
            for k, c in self.experiment.response_info.items()
            if c["_ert_kind"] == "GenDataConfig"
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

    @lru_cache  # noqa: B019
    def load_responses(self, key: str, realizations: Tuple[int]) -> xr.Dataset:
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

        self._ensure_unified_response_dataset_exists(key)

        ds = self.open_unified_response_dataset(key)
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

        reals_with_responses = self.get_realization_list_with_responses()
        if (
            realization_index is not None
            and realization_index not in reals_with_responses
        ):
            raise IndexError(f"No such realization {realization_index}")

        try:
            df = (
                self.load_responses(
                    "summary",
                    (
                        tuple([realization_index])
                        if realization_index is not None
                        else tuple(reals_with_responses)
                    ),
                )
            ).to_dataframe()
            df = df.unstack(level="name")
            df.columns = [col[1] for col in df.columns.values]
            df.index = df.index.rename(
                {"time": "Date", "realization": "Realization"}
            ).reorder_levels(["Realization", "Date"])
            if keys:
                summary_keys = self.get_summary_keyset()
                summary_keys = sorted(
                    [key for key in keys if key in summary_keys]
                )  # ignore keys that doesn't exist
                return df[summary_keys]
            return df
        except (ValueError, KeyError, FileNotFoundError):
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

        if "realizations" not in dataset.dims:
            dataset = dataset.expand_dims(realizations=[realization], axis=0)

        if self._in_memory_datasets.try_store_in_memory(group, realization, dataset):
            return None

        path = self._realization_dir(realization) / f"{group}.nc"
        path.parent.mkdir(exist_ok=True)

        dataset.to_netcdf(path, engine="scipy")

        if os.path.exists(self._path / f"{group}.nc"):
            # Ideally this should never happen
            # But if it does, it will require a recomputation of the
            # unified dataset, regardless of where this is invoked from
            os.remove(self._path / f"{group}.nc")

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

        if self._in_memory_datasets.try_store_in_memory(group, realization, data):
            return None

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
        """Return a pandas dataframe grouped by observation name, showing the
        observation + std values, and accompanying simulated values per realization.

        * key_index is the "{time}" for summary, "{index},{report_step}" for gen_obs
        * Numbers 0...N correspond to the realization index

        Example:
                                     FOPR                        ...
        key_index  2010-01-10 2010-01-20 2010-01-30  ... 2015-06-03 2015-06-13 ...
        OBS          0.001697   0.007549   0.017537  ...   0.020261   0.019794 ...
        STD          0.100000   0.100000   0.100000  ...   0.100000   0.100000 ...
        0            0.055961   0.059060   0.064338  ...   0.060679   0.061015 ...
        1            0.015983   0.018985   0.024111  ...   0.026680   0.026285 ...
        2            0.000000   0.000000   0.000000  ...   0.000000   0.000000 ...
        3            0.283992   0.290090   0.300513  ...   0.299600   0.299727 ...
        4            0.025097   0.028275   0.033700  ...   0.032258   0.032372 ...

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
        return ds[[*dim_order_of_first_var, *data_vars]]

    def _unify_datasets(
        self,
        groups: List[str],
        concat_dim: Literal["realization", "realizations"],
        delete_after: bool = True,
    ) -> None:
        for group in groups:
            if os.path.exists(self._path / f"{group}.nc"):
                continue

            paths = sorted(self.mount_point.glob(f"realization-*/{group}.nc"))

            datasets_in_memory = self._in_memory_datasets.consume_group_if_exists(group)
            datasets_in_files = [xr.open_dataset(p, engine="scipy") for p in paths]
            all_datasets = [*datasets_in_memory, *datasets_in_files]

            if all_datasets:
                combined = xr.combine_nested(all_datasets, concat_dim=concat_dim)
                combined = self._ensure_correct_coordinate_order(combined)

                if not combined:
                    raise ValueError("Unified dataset somehow ended up empty")

                if len(paths) > 0 and delete_after:
                    for p in paths:
                        os.remove(p)

                combined.to_netcdf(self._path / f"{group}.nc")

    def unify_responses(self, key: Optional[str] = None) -> None:
        if key is None:
            for key in self.experiment.response_info:
                self.unify_responses(key)

        gen_data_keys = {
            k
            for k, c in self.experiment.response_info.items()
            if c["_ert_kind"] == "GenDataConfig"
        }

        if key == "gen_data" or key in gen_data_keys:
            if os.path.exists(self._path / "gen_data.nc"):
                return None

            # If gen data, combine across reals,
            # but also across all name(s) into one gen_data.nc

            files_to_remove = []
            to_concat = []
            for group in gen_data_keys:
                paths = sorted(self.mount_point.glob(f"realization-*/{group}.nc"))
                datasets_in_memory = self._in_memory_datasets.consume_group_if_exists(
                    group
                )
                datasets_in_files = [xr.open_dataset(p, engine="scipy") for p in paths]
                all_datasets = [*datasets_in_memory, *datasets_in_files]

                if all_datasets:
                    ds_for_group = xr.concat(
                        [ds.expand_dims(name=[group], axis=1) for ds in all_datasets],
                        dim="realization",
                    )
                    to_concat.append(ds_for_group)

                    files_to_remove.extend(paths)

            # Ensure deterministic ordering wrt name and real
            if to_concat:
                ds = xr.concat(to_concat, dim="name").sortby(["realization", "name"])
                ds = self._ensure_correct_coordinate_order(ds)
                ds.to_netcdf(self._path / "gen_data.nc", engine="scipy")

                for f in files_to_remove:
                    os.remove(f)

        else:
            # If it is a summary, just combined across reals
            self._unify_datasets(
                (
                    [key]
                    if key is not None
                    else list(self.experiment.response_info.keys())
                ),
                "realization",
            )

    def unify_parameters(self, key: Optional[str] = None) -> None:
        self._unify_datasets(
            [key] if key is not None else list(self.experiment.parameter_info.keys()),
            "realizations",
        )
