from __future__ import annotations

import logging
import time
from collections import UserDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import iterative_ensemble_smoother as ies
import numpy as np
import psutil
import xarray as xr
from iterative_ensemble_smoother.experimental import (
    AdaptiveESMDA,
    ensemble_smoother_update_step_row_scaling,
)

from ert.config import Field, GenKwConfig, SurfaceConfig

from ..config.analysis_module import ESSettings, IESSettings
from . import misfit_preprocessor
from .event import AnalysisEvent, AnalysisStatusEvent, AnalysisTimeEvent
from .row_scaling import RowScaling
from .update import RowScalingParameter

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.analysis.configuration import (
        Observation,
        UpdateConfiguration,
        UpdateStep,
    )
    from ert.storage import EnsembleAccessor, EnsembleReader

_logger = logging.getLogger(__name__)


class ErtAnalysisError(Exception):
    pass


@dataclass
class ObservationAndResponseSnapshot:
    obs_name: str
    obs_coords: dict[str, str]
    obs_val: float
    obs_std: float
    obs_scaling: float
    response_mean: float
    response_std: float
    response_mean_mask: bool
    response_std_mask: bool

    def __post_init__(self) -> None:
        status = "Active"
        if np.isnan(self.response_mean):
            status = f"Deactivated, missing response(es): {self.obs_coords}"
        elif not self.response_std_mask:
            status = (
                f"Deactivated, ensemble std "
                f"({self.response_std:.3f}) > STD_CUTOFF, {self.obs_coords}"
            )
        elif not self.response_mean_mask:
            status = f"Deactivated, outlier, {self.obs_coords}"
        self.status = status


@dataclass
class SmootherSnapshot:
    source_case: str
    target_case: str
    alpha: float
    std_cutoff: float
    global_scaling: float
    update_step_snapshots: Dict[str, List[ObservationAndResponseSnapshot]] = field(
        default_factory=dict
    )


def noop_progress_callback(_: AnalysisEvent) -> None:
    pass


class TimedIterator:
    def __init__(
        self, iterable: Sequence[Any], callback: Callable[[AnalysisEvent], None]
    ) -> None:
        self._start_time: float = time.perf_counter()
        self._iterable: Sequence[Any] = iterable
        self._callback: Callable[[AnalysisEvent], None] = callback
        self._index: int = 0

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        try:
            result = self._iterable[self._index]
        except IndexError as e:
            raise StopIteration from e

        if self._index != 0:
            elapsed_time = time.perf_counter() - self._start_time
            estimated_remaining_time = (elapsed_time / (self._index)) * (
                len(self._iterable) - self._index
            )
            self._callback(
                AnalysisTimeEvent(
                    remaining_time=estimated_remaining_time, elapsed_time=elapsed_time
                )
            )

        self._index += 1
        return result


@dataclass
class UpdateSettings:
    std_cutoff: float = 1e-6
    alpha: float = 3.0
    misfit_preprocess: bool = False
    min_required_realizations: int = 2


class TempStorage(UserDict):  # type: ignore
    def __getitem__(self, key: str) -> npt.NDArray[np.double]:
        value: Union[npt.NDArray[np.double], xr.DataArray] = self.data[key]
        if not isinstance(value, xr.DataArray):
            return value
        ensemble_size = len(value.realizations)
        return value.values.reshape(ensemble_size, -1).T

    def __setitem__(
        self, key: str, value: Union[npt.NDArray[np.double], xr.DataArray]
    ) -> None:
        old_value = self.data.get(key)
        if isinstance(old_value, xr.DataArray):
            old_value.data = value.T.reshape(*old_value.shape)
            self.data[key] = old_value
        else:
            self.data[key] = value

    def get_xr_array(self, key: str, real: int) -> xr.DataArray:
        value = self.data[key]
        if isinstance(value, xr.DataArray):
            return value[real]
        else:
            raise ValueError(f"TempStorage has no xarray DataFrame with key={key}")


def _all_parameters(
    source_fs: EnsembleReader,
    iens_active_index: npt.NDArray[np.int_],
    param_groups: List[str],
) -> Optional[npt.NDArray[np.double]]:
    """Return all parameters in assimilation problem"""

    temp_storage = TempStorage()
    for param_group in param_groups:
        _temp_storage = _create_temporary_parameter_storage(
            source_fs, iens_active_index, param_group
        )
        temp_storage[param_group] = _temp_storage[param_group]
    matrices = [temp_storage[p] for p in param_groups]
    return np.vstack(matrices) if matrices else None


def _get_param_with_row_scaling(
    temp_storage: TempStorage,
    parameter: RowScalingParameter,
) -> List[Tuple[npt.NDArray[np.double], RowScaling]]:
    """The row-scaling functionality is implemented in C++ and is made
    accessible through the pybind11 library.
    pybind11 requires that numpy arrays passed to it are in
    Fortran-contiguous order (column-major), which is different from
    numpy's default row-major (C-contiguous) order.
    To ensure compatibility, numpy arrays are explicitly converted to Fortran order.
    It's important to note that if an array originally in C-contiguous
    order is passed to a function expecting Fortran order,
    pybind11 will automatically create a Fortran-ordered copy of the array.
    """
    matrices = []

    if parameter.index_list is None:
        matrices.append(
            (
                np.asfortranarray(temp_storage[parameter.name].astype(np.double)),
                parameter.row_scaling,
            )
        )
    else:
        matrices.append(
            (
                np.asfortranarray(
                    temp_storage[parameter.name][parameter.index_list, :].astype(
                        np.double
                    )
                ),
                parameter.row_scaling,
            ),
        )

    return matrices


def _save_to_temp_storage(
    temp_storage: TempStorage,
    parameter: RowScalingParameter,
    A: Optional[npt.NDArray[np.double]],
) -> None:
    if A is None:
        return
    active_indices = parameter.index_list
    if active_indices is None:
        temp_storage[parameter.name] = A
    else:
        temp_storage[parameter.name][active_indices, :] = A[active_indices, :]


def _save_temp_storage_to_disk(
    target_fs: EnsembleAccessor,
    temp_storage: TempStorage,
    iens_active_index: npt.NDArray[np.int_],
) -> None:
    for key, matrix in temp_storage.items():
        config_node = target_fs.experiment.parameter_configuration[key]
        for i, realization in enumerate(iens_active_index):
            if isinstance(config_node, GenKwConfig):
                assert isinstance(matrix, np.ndarray)
                dataset = xr.Dataset(
                    {
                        "values": ("names", matrix[:, i]),
                        "transformed_values": (
                            "names",
                            config_node.transform(matrix[:, i]),
                        ),
                        "names": [e.name for e in config_node.transfer_functions],
                    }
                )
                target_fs.save_parameters(key, realization, dataset)
            elif isinstance(config_node, (Field, SurfaceConfig)):
                _matrix = temp_storage.get_xr_array(key, i)
                assert isinstance(_matrix, xr.DataArray)
                target_fs.save_parameters(key, realization, _matrix.to_dataset())
            else:
                raise NotImplementedError(f"{type(config_node)} is not supported")


def _create_temporary_parameter_storage(
    source_fs: Union[EnsembleReader, EnsembleAccessor],
    iens_active_index: npt.NDArray[np.int_],
    param_group: str,
) -> TempStorage:
    temp_storage = TempStorage()
    t_genkw = 0.0
    t_surface = 0.0
    t_field = 0.0
    _logger.debug("_create_temporary_parameter_storage() - start")
    config_node = source_fs.experiment.parameter_configuration[param_group]
    matrix: Union[npt.NDArray[np.double], xr.DataArray]
    if isinstance(config_node, GenKwConfig):
        t = time.perf_counter()
        matrix = source_fs.load_parameters(param_group, iens_active_index)[
            "values"
        ].values.T
        t_genkw += time.perf_counter() - t
    elif isinstance(config_node, SurfaceConfig):
        t = time.perf_counter()
        matrix = source_fs.load_parameters(param_group, iens_active_index)["values"]
        t_surface += time.perf_counter() - t
    elif isinstance(config_node, Field):
        t = time.perf_counter()
        matrix = source_fs.load_parameters(param_group, iens_active_index)["values"]
        t_field += time.perf_counter() - t
    else:
        raise NotImplementedError(f"{type(config_node)} is not supported")
    temp_storage[param_group] = matrix
    _logger.debug(
        f"_create_temporary_parameter_storage() time_used gen_kw={t_genkw:.4f}s, \
                surface={t_surface:.4f}s, field={t_field:.4f}s"
    )
    return temp_storage


def _get_obs_and_measure_data(
    source_fs: EnsembleReader,
    selected_observations: List[Observation],
    iens_active_index: npt.NDArray[np.int_],
) -> Tuple[
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.str_],
]:
    measured_data = []
    observation_keys = []
    observation_values = []
    observation_errors = []
    observations = source_fs.experiment.observations
    for obs in selected_observations:
        observation = observations[obs.name]
        group = observation.attrs["response"]
        if obs.index_list:
            index = observation.coords.to_index()[obs.index_list]
            sub_selection = {
                name: list(set(index.get_level_values(name))) for name in index.names
            }
            observation = observation.sel(sub_selection)
        response = source_fs.load_responses(group, tuple(iens_active_index))
        if "time" in observation.coords:
            response = response.reindex(
                time=observation.time, method="nearest", tolerance="1s"  # type: ignore
            )
        try:
            filtered_response = observation.merge(response, join="left")
        except KeyError as e:
            raise ErtAnalysisError(
                f"Mismatched index for: "
                f"Observation: {obs.name} attached to response: {group}"
            ) from e

        obs_coord_names = filtered_response.observations.coords.to_index().names
        observation_keys.append(
            [
                (
                    obs,
                    {
                        coord_name: coords[i]
                        for i, coord_name in enumerate(obs_coord_names)
                    },
                )
                for coords in filtered_response.observations.stack(z=obs_coord_names)
                .coords["z"]
                .data
            ]
        )

        observation_values.append(filtered_response["observations"].data.ravel())
        observation_errors.append(filtered_response["std"].data.ravel())
        measured_data.append(
            filtered_response["values"]
            .transpose(..., "realization")
            .values.reshape((-1, len(filtered_response.realization)))
        )
    source_fs.load_responses.cache_clear()
    return (
        np.concatenate(measured_data, axis=0),
        np.concatenate(observation_values),
        np.concatenate(observation_errors),
        np.concatenate(observation_keys),
    )


def _load_observations_and_responses(
    source_fs: EnsembleReader,
    alpha: float,
    std_cutoff: float,
    global_std_scaling: float,
    iens_ative_index: npt.NDArray[np.int_],
    selected_observations: List[Observation],
    misfit_process: bool,
    update_step_name: str,
) -> Tuple[
    npt.NDArray[np.float_],
    Tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        List[ObservationAndResponseSnapshot],
    ],
]:
    S, observations, errors, obs_keys = _get_obs_and_measure_data(
        source_fs,
        selected_observations,
        iens_ative_index,
    )

    # Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
    # in for example evensen2018 - Analysis of iterative ensemble smoothers for
    # solving inverse problems.
    # `global_std_scaling` is 1.0 for ES.
    scaling = np.sqrt(global_std_scaling) * np.ones_like(errors)
    scaled_errors = errors * scaling

    # Identifies non-outlier observations based on responses.
    ens_mean = S.mean(axis=1)
    ens_std = S.std(ddof=0, axis=1)
    ens_std_mask = ens_std > std_cutoff
    ens_mean_mask = abs(observations - ens_mean) <= alpha * (ens_std + scaled_errors)
    obs_mask = np.logical_and(ens_mean_mask, ens_std_mask)

    if misfit_process:
        scaling[obs_mask] *= misfit_preprocessor.main(
            S[obs_mask], scaled_errors[obs_mask]
        )

    update_snapshot = []
    for (
        (obs_name, obs_coords),
        obs_val,
        obs_std,
        obs_scaling,
        response_mean,
        response_std,
        response_mean_mask,
        response_std_mask,
    ) in zip(
        obs_keys,
        observations,
        errors,
        scaling,
        ens_mean,
        ens_std,
        ens_mean_mask,
        ens_std_mask,
    ):
        update_snapshot.append(
            ObservationAndResponseSnapshot(
                obs_name=obs_name,
                obs_coords=obs_coords,
                obs_val=obs_val,
                obs_std=obs_std,
                obs_scaling=obs_scaling,
                response_mean=response_mean,
                response_std=response_std,
                response_mean_mask=response_mean_mask,
                response_std_mask=response_std_mask,
            )
        )

    for missing_obs in obs_keys[~obs_mask]:
        _logger.warning(f"Deactivating observation: {missing_obs}")

    if len(observations[obs_mask]) == 0:
        raise ErtAnalysisError(
            f"No active observations for update step: {update_step_name}"
        )

    return S[obs_mask], (
        observations[obs_mask],
        scaled_errors[obs_mask],
        update_snapshot,
    )


def _split_by_batchsize(
    arr: npt.NDArray[np.int_], batch_size: int
) -> List[npt.NDArray[np.int_]]:
    """
    Splits an array into sub-arrays of a specified batch size.

    Examples
    --------
    >>> num_params = 10
    >>> batch_size = 3
    >>> s = np.arange(0, num_params)
    >>> _split_by_batchsize(s, batch_size)
    [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]

    >>> num_params = 10
    >>> batch_size = 10
    >>> s = np.arange(0, num_params)
    >>> _split_by_batchsize(s, batch_size)
    [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]

    >>> num_params = 10
    >>> batch_size = 20
    >>> s = np.arange(0, num_params)
    >>> _split_by_batchsize(s, batch_size)
    [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    """
    sections = 1 if batch_size > len(arr) else len(arr) // batch_size
    return np.array_split(arr, sections)


def _update_with_row_scaling(
    update_step: UpdateStep,
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    iens_active_index: npt.NDArray[np.int_],
    S: npt.NDArray[np.float_],
    observation_errors: npt.NDArray[np.float_],
    observation_values: npt.NDArray[np.float_],
    truncation: float,
    inversion_type: str,
    progress_callback: Callable[[AnalysisEvent], None],
    rng: Optional[np.random.Generator] = None,
) -> None:
    for param_group in update_step.row_scaling_parameters:
        source: Union[EnsembleReader, EnsembleAccessor]
        if target_fs.has_parameter_group(param_group.name):
            source = target_fs
        else:
            source = source_fs
        temp_storage = _create_temporary_parameter_storage(
            source, iens_active_index, param_group.name
        )

        # Initialize and run a smoother with row scaling
        params_with_row_scaling = ensemble_smoother_update_step_row_scaling(
            covariance=observation_errors**2,
            observations=observation_values,
            X_with_row_scaling=_get_param_with_row_scaling(temp_storage, param_group),
            Y=S,
            inversion=inversion_type,
            truncation=truncation,
            seed=rng,
        )

        # Store result
        _save_to_temp_storage(temp_storage, param_group, params_with_row_scaling[0][0])
        progress_callback(
            AnalysisStatusEvent(msg=f"Storing data for {param_group.name}..")
        )
        _save_temp_storage_to_disk(target_fs, temp_storage, iens_active_index)


def _copy_unupdated_parameters(
    all_parameter_groups: List[str],
    updated_parameter_groups: List[str],
    iens_active_index: npt.NDArray[np.int_],
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
) -> None:
    """
    Copies parameter groups that have not been updated from a source ensemble to a target ensemble.
    This function ensures that all realizations in the target ensemble have a complete set of parameters,
    including those that were not updated.
    This is necessary because users can choose not to update parameters but may still want to analyse them.

    Parameters:
    all_parameter_groups (List[str]): A list of all parameter groups.
    updated_parameter_groups (List[str]): A list of parameter groups that have already been updated.
    iens_active_index (npt.NDArray[np.int_]): An array of indices for the active realizations in the
                                              target ensemble.
    source_fs (EnsembleReader): The file system of the source ensemble, from which parameters are copied.
    target_fs (EnsembleAccessor): The file system of the target ensemble, to which parameters are saved.

    Returns:
    None: The function does not return any value but updates the target file system by copying over
    the parameters.
    """
    # Identify parameter groups that have not been updated
    not_updated_parameter_groups = list(
        set(all_parameter_groups) - set(updated_parameter_groups)
    )

    # Copy the non-updated parameter groups from source to target for each active realization
    for parameter_group in not_updated_parameter_groups:
        for realization in iens_active_index:
            ds = source_fs.load_parameters(parameter_group, int(realization))
            target_fs.save_parameters(parameter_group, realization, ds)


def analysis_ES(
    updatestep: UpdateConfiguration,
    rng: np.random.Generator,
    module: ESSettings,
    alpha: float,
    std_cutoff: float,
    global_scaling: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    progress_callback: Callable[[AnalysisEvent], None],
    misfit_process: bool,
) -> None:
    iens_active_index = np.flatnonzero(ens_mask)

    ensemble_size = ens_mask.sum()
    updated_parameter_groups = []

    for update_step in updatestep:
        updated_parameter_groups.extend(
            [param_group.name for param_group in update_step.parameters]
        )

        progress_callback(
            AnalysisStatusEvent(msg="Loading observations and responses..")
        )
        (
            S,
            (
                observation_values,
                observation_errors,
                update_snapshot,
            ),
        ) = _load_observations_and_responses(
            source_fs,
            alpha,
            std_cutoff,
            global_scaling,
            iens_active_index,
            update_step.observations,
            misfit_process,
            update_step.name,
        )

        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot

        num_obs = len(observation_values)

        smoother_es = ies.ESMDA(
            covariance=observation_errors**2,
            observations=observation_values,
            alpha=1,  # The user is responsible for scaling observation covariance (esmda usage)
            seed=rng,
            inversion=module.inversion,
        )
        truncation = module.enkf_truncation

        if module.localization:
            smoother_adaptive_es = AdaptiveESMDA(
                covariance=observation_errors**2,
                observations=observation_values,
                seed=rng,
            )

            # Pre-calculate cov_YY
            cov_YY = np.cov(S)

            D = smoother_adaptive_es.perturb_observations(
                ensemble_size=ensemble_size, alpha=1.0
            )

        else:
            # Compute transition matrix so that
            # X_posterior = X_prior @ T
            T = smoother_es.compute_transition_matrix(
                Y=S, alpha=1.0, truncation=truncation
            )
            # Add identity in place for fast computation
            np.fill_diagonal(T, T.diagonal() + 1)

        for param_group in update_step.parameters:
            source: Union[EnsembleReader, EnsembleAccessor]
            if target_fs.has_parameter_group(param_group.name):
                source = target_fs
            else:
                source = source_fs
            temp_storage = _create_temporary_parameter_storage(
                source, iens_active_index, param_group.name
            )
            if module.localization:
                num_params = temp_storage[param_group.name].shape[0]

                # Calculate adaptive batch size.
                # Adaptive Localization calculates the cross-covariance between
                # parameters and responses.
                # Cross-covariance is a matrix with shape num_params x num_obs
                # which may be larger than memory.

                # From `psutil` documentation:
                # - available:
                # the memory that can be given instantly to processes without the
                # system going into swap.
                # This is calculated by summing different memory values depending
                # on the platform and it is supposed to be used to monitor actual
                # memory usage in a cross platform fashion.
                available_memory_bytes = psutil.virtual_memory().available
                memory_safety_factor = 0.8
                bytes_in_float64 = 8
                batch_size = min(
                    int(
                        np.floor(
                            available_memory_bytes
                            * memory_safety_factor
                            / (num_obs * bytes_in_float64)
                        )
                    ),
                    num_params,
                )

                batches = _split_by_batchsize(np.arange(0, num_params), batch_size)

                log_msg = f"Running localization on {num_params} parameters, {num_obs} responses, {ensemble_size} realizations and {len(batches)} batches"
                _logger.info(log_msg)
                progress_callback(AnalysisStatusEvent(msg=log_msg))

                start = time.time()
                for param_batch_idx in TimedIterator(batches, progress_callback):
                    X_local = temp_storage[param_group.name][param_batch_idx, :]
                    temp_storage[param_group.name][
                        param_batch_idx, :
                    ] = smoother_adaptive_es.assimilate(
                        X=X_local,
                        Y=S,
                        D=D,
                        alpha=1.0,  # The user is responsible for scaling observation covariance (esmda usage)
                        correlation_threshold=module.correlation_threshold,
                        cov_YY=cov_YY,
                        verbose=False,
                    )
                _logger.info(
                    f"Adaptive Localization of {param_group} completed in {(time.time() - start) / 60} minutes"
                )

            else:
                # Use low-level ies API to allow looping over parameters
                if active_indices := param_group.index_list:
                    # The batch of parameters
                    X_local = temp_storage[param_group.name][active_indices, :]

                    # Update manually using global transition matrix T
                    temp_storage[param_group.name][active_indices, :] = X_local @ T

                else:
                    # The batch of parameters
                    X_local = temp_storage[param_group.name]

                    # Update manually using global transition matrix T
                    temp_storage[param_group.name] = X_local @ T

            log_msg = f"Storing data for {param_group.name}.."
            _logger.info(log_msg)
            progress_callback(AnalysisStatusEvent(msg=log_msg))
            start = time.time()
            _save_temp_storage_to_disk(target_fs, temp_storage, iens_active_index)
            _logger.info(
                f"Storing data for {param_group.name} completed in {(time.time() - start) / 60} minutes"
            )

        _copy_unupdated_parameters(
            list(source_fs.experiment.parameter_configuration.keys()),
            updated_parameter_groups,
            iens_active_index,
            source_fs,
            target_fs,
        )

        _update_with_row_scaling(
            update_step=update_step,
            source_fs=source_fs,
            target_fs=target_fs,
            iens_active_index=iens_active_index,
            S=S,
            observation_errors=observation_errors,
            observation_values=observation_values,
            truncation=truncation,
            inversion_type=module.inversion,
            progress_callback=progress_callback,
            rng=rng,
        )


def analysis_IES(
    update_config: UpdateConfiguration,
    rng: np.random.Generator,
    analysis_config: IESSettings,
    alpha: float,
    std_cutoff: float,
    smoother_snapshot: SmootherSnapshot,
    ens_mask: npt.NDArray[np.bool_],
    source_fs: EnsembleReader,
    target_fs: EnsembleAccessor,
    sies_smoother: Optional[ies.SIES],
    progress_callback: Callable[[AnalysisEvent], None],
    misfit_preprocessor: bool,
    sies_step_length: Callable[[int], float],
    initial_mask: npt.NDArray[np.bool_],
) -> ies.SIES:
    iens_active_index = np.flatnonzero(ens_mask)
    updated_parameter_groups = []
    # Pick out realizations that were among the initials that are still living
    # Example: initial_mask=[1,1,1,0,1], ens_mask=[0,1,1,0,1]
    # Then the result is [0,1,1,1]
    # This is needed for the SIES library
    masking_of_initial_parameters = ens_mask[initial_mask]

    # It is not the iterations relating to IES or ESMDA.
    # It is related to functionality for turning on/off groups of parameters and observations.
    for update_step in update_config:
        updated_parameter_groups.extend(
            [param_group.name for param_group in update_step.parameters]
        )

        progress_callback(
            AnalysisStatusEvent(msg="Loading observations and responses..")
        )

        (
            S,
            (
                observation_values,
                observation_errors,
                update_snapshot,
            ),
        ) = _load_observations_and_responses(
            source_fs,
            alpha,
            std_cutoff,
            1.0,
            iens_active_index,
            update_step.observations,
            misfit_preprocessor,
            update_step.name,
        )

        smoother_snapshot.update_step_snapshots[update_step.name] = update_snapshot
        if len(observation_values) == 0:
            raise ErtAnalysisError(
                f"No active observations for update step: {update_step.name}."
            )

        # if the algorithm object is not passed, initialize it
        if sies_smoother is None:
            # The sies smoother must be initialized with the full parameter ensemble
            # Get relevant active realizations
            param_groups = list(source_fs.experiment.parameter_configuration.keys())
            parameter_ensemble_active = _all_parameters(
                source_fs, iens_active_index, param_groups
            )
            sies_smoother = ies.SIES(
                parameters=parameter_ensemble_active,
                covariance=observation_errors**2,
                observations=observation_values,
                seed=rng,
                inversion=analysis_config.inversion,
                truncation=analysis_config.enkf_truncation,
            )

            # Keep track of iterations to calculate step-lengths
            sies_smoother.iteration = 1

        # Calculate step-lengths to scale SIES iteration
        step_length = sies_step_length(sies_smoother.iteration)

        # Propose a transition matrix using only active realizations
        proposed_W = sies_smoother.propose_W_masked(
            S, ensemble_mask=masking_of_initial_parameters, step_length=step_length
        )

        # Store transition matrix for later use on sies object
        sies_smoother.W[:, masking_of_initial_parameters] = proposed_W

        for param_group in update_step.parameters:
            source: Union[EnsembleReader, EnsembleAccessor] = target_fs
            try:
                target_fs.load_parameters(group=param_group.name, realizations=0)[
                    "values"
                ]
            except Exception:
                source = source_fs
            temp_storage = _create_temporary_parameter_storage(
                source, iens_active_index, param_group.name
            )
            if active_parameter_indices := param_group.index_list:
                X = temp_storage[param_group.name][active_parameter_indices, :]
                temp_storage[param_group.name][
                    active_parameter_indices, :
                ] = X + X @ sies_smoother.W / np.sqrt(len(iens_active_index) - 1)
            else:
                X = temp_storage[param_group.name]
                temp_storage[param_group.name] = X + X @ sies_smoother.W / np.sqrt(
                    len(iens_active_index) - 1
                )

            progress_callback(
                AnalysisStatusEvent(msg=f"Storing data for {param_group.name}..")
            )
            _save_temp_storage_to_disk(target_fs, temp_storage, iens_active_index)

        _copy_unupdated_parameters(
            list(source_fs.experiment.parameter_configuration.keys()),
            updated_parameter_groups,
            iens_active_index,
            source_fs,
            target_fs,
        )

    assert sies_smoother is not None, "sies_smoother should be initialized"

    # Increment the iteration number
    sies_smoother.iteration += 1

    # Return the sies smoother so it may be iterated over
    return sies_smoother


def _write_update_report(path: Path, snapshot: SmootherSnapshot, run_id: str) -> None:
    fname = path / f"{run_id}.txt"
    fname.parent.mkdir(parents=True, exist_ok=True)
    for update_step_name, update_step in snapshot.update_step_snapshots.items():
        with open(fname, "a", encoding="utf-8") as fout:
            fout.write("=" * 150 + "\n")
            timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
            fout.write(f"Time: {timestamp}\n")
            fout.write(f"Parent ensemble: {snapshot.source_case}\n")
            fout.write(f"Target ensemble: {snapshot.target_case}\n")
            fout.write(f"Alpha: {snapshot.alpha}\n")
            fout.write(f"Global scaling: {snapshot.global_scaling}\n")
            fout.write(f"Standard cutoff: {snapshot.std_cutoff}\n")
            fout.write(f"Run id: {run_id}\n")
            fout.write(f"Update step: {update_step_name:<10}\n")
            fout.write("-" * 150 + "\n")
            fout.write(
                "Observed history".rjust(56)
                + "|".rjust(17)
                + "Simulated data".rjust(32)
                + "|".rjust(13)
                + "Status".rjust(12)
                + "\n"
            )
            fout.write("-" * 150 + "\n")
            for nr, step in enumerate(update_step):
                obs_std = (
                    f"{step.obs_std:.3f}"
                    if step.obs_scaling == 1
                    else f"{step.obs_std * step.obs_scaling:.3f} ({step.obs_std:<.3f} * {step.obs_scaling:.3f})"
                )
                fout.write(
                    f"{nr+1:^6}: {step.obs_name:20} {step.obs_val:>16.3f} +/- "
                    f"{obs_std:<21} | {step.response_mean:>21.3f} +/- "
                    f"{step.response_std:<16.3f} {'|':<6} "
                    f"{step.status.capitalize()}\n"
                )


def _assert_has_enough_realizations(
    ens_mask: npt.NDArray[np.bool_], analysis_config: UpdateSettings
) -> None:
    active_realizations = ens_mask.sum()
    if active_realizations < analysis_config.min_required_realizations:
        raise ErtAnalysisError(
            f"There are {active_realizations} active realisations left, which is "
            "less than the minimum specified - stopping assimilation.",
        )


def _create_smoother_snapshot(
    prior_name: str,
    posterior_name: str,
    analysis_config: UpdateSettings,
    global_scaling: float,
) -> SmootherSnapshot:
    return SmootherSnapshot(
        prior_name,
        posterior_name,
        analysis_config.alpha,
        analysis_config.std_cutoff,
        global_scaling=global_scaling,
    )


def smoother_update(
    prior_storage: EnsembleReader,
    posterior_storage: EnsembleAccessor,
    run_id: str,
    updatestep: UpdateConfiguration,
    analysis_config: Optional[UpdateSettings] = None,
    es_settings: Optional[ESSettings] = None,
    rng: Optional[np.random.Generator] = None,
    progress_callback: Optional[Callable[[AnalysisEvent], None]] = None,
    global_scaling: float = 1.0,
    log_path: Optional[Path] = None,
) -> SmootherSnapshot:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if rng is None:
        rng = np.random.default_rng()
    analysis_config = UpdateSettings() if analysis_config is None else analysis_config
    es_settings = ESSettings() if es_settings is None else es_settings
    ens_mask = prior_storage.get_realization_mask_with_responses()
    _assert_has_enough_realizations(ens_mask, analysis_config)

    smoother_snapshot = _create_smoother_snapshot(
        prior_storage.name,
        posterior_storage.name,
        analysis_config,
        global_scaling,
    )

    analysis_ES(
        updatestep,
        rng,
        es_settings,
        analysis_config.alpha,
        analysis_config.std_cutoff,
        global_scaling,
        smoother_snapshot,
        ens_mask,
        prior_storage,
        posterior_storage,
        progress_callback,
        analysis_config.misfit_preprocess,
    )

    if log_path is not None:
        _write_update_report(
            log_path,
            smoother_snapshot,
            run_id,
        )

    return smoother_snapshot


def iterative_smoother_update(
    prior_storage: EnsembleReader,
    posterior_storage: EnsembleAccessor,
    sies_smoother: Optional[ies.SIES],
    run_id: str,
    update_config: UpdateConfiguration,
    update_settings: UpdateSettings,
    analysis_config: IESSettings,
    sies_step_length: Callable[[int], float],
    initial_mask: npt.NDArray[np.bool_],
    rng: Optional[np.random.Generator] = None,
    progress_callback: Optional[Callable[[AnalysisEvent], None]] = None,
    log_path: Optional[Path] = None,
    global_scaling: float = 1.0,
) -> Tuple[SmootherSnapshot, ies.SIES]:
    if not progress_callback:
        progress_callback = noop_progress_callback
    if rng is None:
        rng = np.random.default_rng()

    if len(update_config) > 1:
        raise ErtAnalysisError(
            "Can not combine IES_ENKF modules with multi step updates"
        )

    ens_mask = prior_storage.get_realization_mask_with_responses()
    _assert_has_enough_realizations(ens_mask, update_settings)

    smoother_snapshot = _create_smoother_snapshot(
        prior_storage.name,
        posterior_storage.name,
        update_settings,
        global_scaling,
    )

    sies_smoother = analysis_IES(
        update_config=update_config,
        rng=rng,
        analysis_config=analysis_config,
        alpha=update_settings.alpha,
        std_cutoff=update_settings.std_cutoff,
        smoother_snapshot=smoother_snapshot,
        ens_mask=ens_mask,
        source_fs=prior_storage,
        target_fs=posterior_storage,
        sies_smoother=sies_smoother,
        progress_callback=progress_callback,
        misfit_preprocessor=update_settings.misfit_preprocess,
        sies_step_length=sies_step_length,
        initial_mask=initial_mask,
    )
    if log_path is not None:
        _write_update_report(
            log_path,
            smoother_snapshot,
            run_id,
        )

    return smoother_snapshot, sies_smoother
