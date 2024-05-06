from __future__ import annotations

import json
import logging
import time
from multiprocessing.pool import ThreadPool
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from pandas import DataFrame

from ert.analysis import AnalysisEvent, SmootherSnapshot, smoother_update
from ert.callbacks import forward_model_ok
from ert.config import (
    ErtConfig,
    Field,
    GenKwConfig,
)
from ert.config.observations import EnkfObs
from ert.data import MeasuredData
from ert.data._measured_data import ObservationError, ResponseError
from ert.load_status import LoadResult, LoadStatus

from .enkf_main import EnKFMain, ensemble_context

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import (
        PriorDict,
        WorkflowJob,
    )
    from ert.run_arg import RunArg
    from ert.storage import Ensemble, Storage


def _load_realization(
    realisation: int,
    run_args: List[RunArg],
) -> Tuple[LoadResult, int]:
    result = forward_model_ok(run_args[realisation])
    return result, realisation


class LibresFacade:
    """The intention of this class is to expose properties or data of ert
    commonly used in other project. It is part of the public interface of ert,
    and as such changes here should not be taken lightly."""

    def __init__(self, enkf_main: Union[EnKFMain, ErtConfig]):
        # EnKFMain is more or less just a facade for the configuration at this
        # point, so in the process of removing it altogether it is easier
        # if we allow the facade to created with both EnKFMain and ErtConfig
        if isinstance(enkf_main, EnKFMain):
            self._enkf_main = enkf_main
            self.config: ErtConfig = enkf_main.ert_config
        else:
            self._enkf_main = EnKFMain(enkf_main)
            self.config = enkf_main
        self.update_snapshots: Dict[str, SmootherSnapshot] = {}
        self.update_configuration = None

    def smoother_update(
        self,
        prior_storage: Ensemble,
        posterior_storage: Ensemble,
        run_id: str,
        observations: Iterable[str],
        parameters: Iterable[str],
        progress_callback: Optional[Callable[[AnalysisEvent], None]] = None,
        global_std_scaling: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        misfit_process: bool = False,
    ) -> SmootherSnapshot:
        if rng is None:
            rng = np.random.default_rng()
        update_snapshot = smoother_update(
            prior_storage,
            posterior_storage,
            run_id,
            observations,
            parameters,
            self.config.analysis_config.observation_settings,
            self.config.analysis_config.es_module,
            rng,
            progress_callback,
            global_std_scaling,
            log_path=self.config.analysis_config.log_path,
        )
        self.update_snapshots[run_id] = update_snapshot
        return update_snapshot

    @property
    def enspath(self) -> str:
        return self.config.ens_path

    @property
    def user_config_file(self) -> Optional[str]:
        return self.config.user_config_file

    def get_field_parameters(self) -> List[str]:
        return list(
            val.name
            for val in self.config.ensemble_config.parameter_configuration
            if isinstance(val, Field)
        )

    def get_gen_kw(self) -> List[str]:
        return self.config.ensemble_config.get_keylist_gen_kw()

    @property
    def grid_file(self) -> Optional[str]:
        return self.config.ensemble_config.grid_file

    def get_ensemble_size(self) -> int:
        return self.config.model_config.num_realizations

    @property
    def run_path(self) -> str:
        return self.config.model_config.runpath_format_string

    def load_from_forward_model(
        self,
        ensemble: Ensemble,
        realisations: npt.NDArray[np.bool_],
        iteration: int,
    ) -> int:
        t = time.perf_counter()
        run_context = ensemble_context(
            ensemble,
            realisations,
            iteration,
            self.config.substitution_list,
            jobname_format=self.config.model_config.jobname_format_string,
            runpath_format=self.config.model_config.runpath_format_string,
            runpath_file=self.config.runpath_file,
        )

        nr_loaded = self._load_from_run_path(
            self.config.model_config.num_realizations,
            run_context.run_args,
            run_context.mask,
        )
        _logger.debug(
            f"load_from_forward_model() time_used {(time.perf_counter() - t):.4f}s"
        )
        return nr_loaded

    @staticmethod
    def _load_from_run_path(
        ensemble_size: int,
        run_args: List[RunArg],
        active_realizations: List[bool],
    ) -> int:
        """Returns the number of loaded realizations"""
        pool = ThreadPool(processes=8)

        async_result = [
            pool.apply_async(
                _load_realization,
                (iens, run_args),
            )
            for iens in range(ensemble_size)
            if active_realizations[iens]
        ]

        loaded = 0
        for t in async_result:
            ((status, message), iens) = t.get()

            if status == LoadStatus.LOAD_SUCCESSFUL:
                loaded += 1
            else:
                _logger.error(f"Realization: {iens}, load failure: {message}")

        return loaded

    def get_observations(self) -> "EnkfObs":
        return self.config.observations

    def _get_response_name_for_obs_name(self, observation_name: str) -> str:
        obs_ds = next(
            ds.sel(obs_name=observation_name, drop=True)
            for ds in self.config.observations.datasets.values()
            if observation_name in ds["obs_name"]
        )

        if obs_ds is None:
            all_obs_and_response_keys = {
                k: ds["obs_name"].data.tolist()
                for k, ds in self.config.observations.datasets.items()
            }
            raise KeyError(
                f"Did not find observation {observation_name} in "
                "any observation datasets. All observations: "
                f"{json.dumps(all_obs_and_response_keys)}"
            )

        return str(obs_ds["name"].data.tolist()[0])

    def get_data_key_for_obs_key(self, observation_key: str) -> str:
        return self._get_response_name_for_obs_name(observation_key)

    @staticmethod
    def load_all_misfit_data(ensemble: Ensemble) -> DataFrame:
        """Loads all misfit data for a given ensemble.

        Retrieves all active realizations from the ensemble, and for each
        realization, it gathers the observations and measured data. The
        function then calculates the misfit, which is a measure of the
        discrepancy between observed and simulated values, for each data
        column. The misfit is calculated as the squared difference between the
        observed and measured data, normalized by the standard deviation of the
        observations.

        The misfit data is then grouped by key, summed, and transposed to form
        a DataFrame. The DataFrame has an additional column "MISFIT:TOTAL",
        which is the sum of all misfits for each realization. The index of the
        DataFrame is named "Realization".

        Parameters:
            ensemble: The ensemble from which to load the misfit data.

        Returns:
            DataFrame: A DataFrame containing the misfit data for all
                realizations in the ensemble. Each column (except for "MISFIT:TOTAL")
                corresponds to a key in the measured data, and each row corresponds
                to a realization. The "MISFIT:TOTAL" column contains the total
                misfit for each realization.
        """
        try:
            measured_data = MeasuredData(ensemble)
        except (ResponseError, ObservationError):
            return DataFrame()
        misfit = DataFrame()
        for name in measured_data.data.columns.unique(0):
            df = (
                (
                    measured_data.data[name].loc["OBS"]
                    - measured_data.get_simulated_data()[name]
                )
                / measured_data.data[name].loc["STD"]
            ) ** 2
            misfit[f"MISFIT:{name}"] = df.sum(axis=1)
        misfit["MISFIT:TOTAL"] = misfit.sum(axis=1)
        misfit.index.name = "Realization"
        misfit.index = misfit.index.astype(int)

        return misfit

    def get_summary_keys(self) -> List[str]:
        return self.config.ensemble_config.get_summary_keys()

    def gen_kw_priors(self) -> Dict[str, List["PriorDict"]]:
        gen_kw_keys = self.get_gen_kw()
        all_gen_kw_priors = {}
        for key in gen_kw_keys:
            gen_kw_config = self.config.ensemble_config.parameter_configs[key]
            if isinstance(gen_kw_config, GenKwConfig):
                all_gen_kw_priors[key] = gen_kw_config.get_priors()

        return all_gen_kw_priors

    def get_workflow_job(self, name: str) -> Optional["WorkflowJob"]:
        return self.config.workflow_jobs.get(name)

    def run_ertscript(  # type: ignore
        self,
        ertscript,
        storage: Storage,
        ensemble: Ensemble,
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> Any:
        return ertscript(self._enkf_main, storage, ensemble=ensemble).run(
            *args, **kwargs
        )

    @classmethod
    def from_config_file(
        cls, config_file: str, read_only: bool = False
    ) -> "LibresFacade":
        return cls(EnKFMain(ErtConfig.from_file(config_file), read_only))
