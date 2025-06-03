from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Any
from uuid import UUID

from pydantic import PrivateAttr

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble

from ..analysis import ErtAnalysisError, enif_update
from ..config import HookRuntime
from ..plugins import PostUpdateFixtures, PreFirstUpdateFixtures, PreUpdateFixtures
from . import RunModelStatusEvent, RunModelUpdateBeginEvent
from .base_run_model import ErtRunError, UpdateRunModel

logger = logging.getLogger(__name__)


class ManualUpdateEnIF(UpdateRunModel):
    ensemble_id: str
    support_restart: bool = False

    _prior: Ensemble = PrivateAttr()

    def model_post_init(self, ctx: Any) -> None:
        super().model_post_init(ctx)

        try:
            self._prior = self._storage.get_ensemble(UUID(self.ensemble_id))
        except (KeyError, ValueError) as err:
            raise ErtRunError(
                f"Prior ensemble with ID: {UUID(self.ensemble_id)} does not exist"
            ) from err

        self.support_restart = False

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        self.set_env_key("_ERT_EXPERIMENT_ID", str(self._prior.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(self._prior.id))

        ensemble_format = self.target_ensemble
        self.update(self._prior, ensemble_format % (self._prior.iteration + 1))

    @classmethod
    def name(cls) -> str:
        return "Manual EnIF update"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → EnIF update"

    def check_if_runpath_exists(self) -> bool:
        # Will not run a forward model, so does not create files on runpath
        return False

    def update(
        self,
        prior: Ensemble,
        posterior_name: str,
        weight: float = 1.0,
    ) -> Ensemble:
        self.validate_successful_realizations_count()
        self.send_event(
            RunModelUpdateBeginEvent(iteration=prior.iteration, run_id=prior.id)
        )
        self.send_event(
            RunModelStatusEvent(
                iteration=prior.iteration,
                run_id=prior.id,
                msg="Creating posterior ensemble..",
            )
        )

        pre_first_update_fixtures = PreFirstUpdateFixtures(
            storage=self._storage,
            ensemble=prior,
            observation_settings=self.update_settings,
            es_settings=self.analysis_settings,
            random_seed=self.random_seed,
            reports_dir=self.reports_dir(experiment_name=prior.experiment.name),
            run_paths=self._run_paths,
        )

        posterior = self._storage.create_ensemble(
            prior.experiment,
            ensemble_size=prior.ensemble_size,
            iteration=prior.iteration + 1,
            name=posterior_name,
            prior_ensemble=prior,
        )
        if prior.iteration == 0:
            self.run_workflows(
                fixtures=pre_first_update_fixtures,
            )

        update_args_dict = {
            field.name: getattr(pre_first_update_fixtures, field.name)
            for field in dataclasses.fields(pre_first_update_fixtures)
        }

        self.run_workflows(
            fixtures=PreUpdateFixtures(
                **{**update_args_dict, "hook": HookRuntime.PRE_UPDATE}
            ),
        )
        try:
            enif_update(
                prior,
                posterior,
                parameters=prior.experiment.update_parameters,
                observations=prior.experiment.observation_keys,
                global_scaling=weight,
                random_seed=self.random_seed,
                progress_callback=functools.partial(
                    self.send_smoother_event,
                    prior.iteration,
                    prior.id,
                ),
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                "Update algorithm failed for iteration:"
                f"{posterior.iteration}. The following error occurred: {e}"
            ) from e

        self.run_workflows(
            fixtures=PostUpdateFixtures(
                **{**update_args_dict, "hook": HookRuntime.POST_UPDATE}
            ),
        )
        return posterior
