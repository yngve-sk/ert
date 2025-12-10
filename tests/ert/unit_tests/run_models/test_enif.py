import queue
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.mode_definitions import (
    ENIF_MODE,
)
from ert.run_models import create_model
from ert.storage import open_storage
from tests.ert.conftest import _create_design_matrix


@pytest.mark.integration_test
def test_that_enif_update_does_not_update_design_matrix_parameters(
    copy_case,
):
    """
    1. Runs EnIF on poly.ert
    2. Runs EnIF on poly.ert with parameter "a" values from first run in design matrix
    3. Expect parameter "a" values to remain unchanged after second run
    """
    num_realizations = 10

    copy_case("poly_example")
    config_file = Path("poly.ert")

    with open(config_file, "a", encoding="utf-8") as fh:
        fh.write(f"NUM_REALIZATIONS {num_realizations}\n")
        fh.write("RANDOM_SEED 123456789\n")

    ert_config_without_dm = ErtConfig.from_file("poly.ert")

    evaluator_server_config = EvaluatorServerConfig()
    enif_without_dm = create_model(
        ert_config_without_dm,
        args=Namespace(
            mode=ENIF_MODE,
            experiment_name="enif_without_dm",
            target_ensemble="ens_without_dm_%d",
        ),
        status_queue=queue.SimpleQueue(),
    )
    enif_without_dm.start_simulations_thread(evaluator_server_config)

    with open_storage(enif_without_dm.storage_path, mode="r") as storage:
        previous_experiment = storage.get_experiment_by_name("enif_without_dm")
        posterior_ensemble = next(
            e for e in previous_experiment.ensembles if e.iteration == 1
        )
        posterior_a_values = posterior_ensemble.load_parameters("a")

        _create_design_matrix(
            "poly_design.xlsx",
            pl.DataFrame(
                {
                    "REAL": list(range(num_realizations)),
                    "a": posterior_a_values["a"].to_list(),
                }
            ),
        )

        with open(config_file, "a", encoding="utf-8") as fh:
            fh.write("DESIGN_MATRIX poly_design.xlsx\n")

    ert_config_with_dm = ErtConfig.from_file("poly.ert")
    assert ert_config_with_dm.random_seed == ert_config_without_dm.random_seed

    enif_with_dm = create_model(
        ert_config_with_dm,
        args=Namespace(
            mode=ENIF_MODE,
            experiment_name="enif_with_dm",
            target_ensemble="ens_with_dm_%d",
        ),
        status_queue=queue.SimpleQueue(),
    )

    assert enif_with_dm.random_seed == enif_without_dm.random_seed
    enif_with_dm.start_simulations_thread(evaluator_server_config)

    with open_storage(enif_with_dm.storage_path, mode="r") as storage:
        experiment_without_dm = storage.get_experiment_by_name("enif_without_dm")
        experiment_with_dm = storage.get_experiment_by_name("enif_with_dm")

        prior_with_dm = experiment_with_dm.get_ensemble_by_name("ens_with_dm_0")
        posterior_with_dm = experiment_with_dm.get_ensemble_by_name("ens_with_dm_1")

        prior_without_dm = experiment_without_dm.get_ensemble_by_name(
            "ens_without_dm_0"
        )
        posterior_without_dm = experiment_without_dm.get_ensemble_by_name(
            "ens_without_dm_1"
        )

        prior_with_dm_a = prior_with_dm.load_parameters("a")["a"]
        posterior_with_dm_a = posterior_with_dm.load_parameters("a")["a"]
        prior_without_dm_a = prior_without_dm.load_parameters("a")["a"]
        posterior_without_dm_a = posterior_without_dm.load_parameters("a")["a"]

        assert posterior_with_dm_a.equals(prior_with_dm_a)

        # Expect less posterior variance when a was updated
        assert prior_without_dm_a.var() > posterior_without_dm_a.var()
