import logging
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from resdata.summary import Summary

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, ensemble_context
from ert.libres_facade import LibresFacade
from ert.storage import open_storage


@pytest.fixture()
def setup_case(storage, use_tmpdir):
    def func(config_text):
        Path("config.ert").write_text(config_text, encoding="utf-8")

        ert_config = ErtConfig.from_file("config.ert")
        prior_ensemble = storage.create_ensemble(
            storage.create_experiment(
                responses=ert_config.ensemble_config.response_configuration
            ),
            name="prior",
            ensemble_size=ert_config.model_config.num_realizations,
        )
        run_context = ensemble_context(
            prior_ensemble,
            [True],
            0,
            None,
            "",
            ert_config.model_config.runpath_format_string,
            "name",
        )
        create_run_path(run_context, ert_config)
        return ert_config, prior_ensemble

    yield func


def run_simulator(time_step_count, start_date) -> Summary:
    summary = Summary.writer("SNAKE_OIL_FIELD", start_date, 10, 10, 10)

    summary.add_variable("FOPR", unit="SM3/DAY")
    summary.add_variable("FOPRH", unit="SM3/DAY")

    summary.add_variable("WOPR", wgname="OP1", unit="SM3/DAY")
    summary.add_variable("WOPRH", wgname="OP1", unit="SM3/DAY")

    mini_step_count = 10
    for report_step in range(time_step_count):
        for mini_step in range(mini_step_count):
            t_step = summary.add_t_step(
                report_step + 1, sim_days=report_step * mini_step_count + mini_step
            )
            t_step["FOPR"] = 1
            t_step["WOPR:OP1"] = 2
            t_step["FOPRH"] = 3
            t_step["WOPRH:OP1"] = 4

    return summary


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_load_forward_model(snake_oil_default_storage):
    """
    Checking that we are able to load from forward model
    """
    facade = LibresFacade.from_config_file("snake_oil.ert")
    realisation_number = 0

    realizations = [False] * facade.get_ensemble_size()
    realizations[realisation_number] = True

    with open_storage(facade.enspath, mode="w") as storage:
        # 'load_from_forward_model' requires the ensemble to be writeable...
        default = storage.get_ensemble_by_name("default_0")

        loaded = facade.load_from_forward_model(default, realizations, 0)
        assert loaded == 1
        assert default.get_realization_mask_with_responses()[
            realisation_number
        ]  # Check that status is as expected


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "summary_configuration, expected",
    [
        pytest.param(
            None,
            (1, None),
            id=(
                "Checking that we are able to successfully load "
                "from forward model with eclipse case even though "
                "there is no eclipse file in the run path. This is "
                "because no SUMMARY is added to the config"
            ),
        ),
        pytest.param(
            "SUMMARY *",
            (0, "Could not find any unified summary file"),
            id=(
                "Check that loading fails if we have configured"
                "SUMMARY but no summary is available in the run path"
            ),
        ),
    ],
)
def test_load_forward_model_summary(summary_configuration, storage, expected, caplog):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE SNAKE_OIL_FIELD
        REFCASE SNAKE_OIL_FIELD
        """
    )
    if summary_configuration:
        config_text += summary_configuration
    Path("config.ert").write_text(config_text, encoding="utf-8")
    # Create refcase
    ecl_sum = run_simulator(1, datetime(2014, 9, 10))
    ecl_sum.fwrite()

    ert_config = ErtConfig.from_file("config.ert")
    experiment_id = storage.create_experiment(
        responses=ert_config.ensemble_config.response_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=100
    )

    run_context = ensemble_context(
        prior_ensemble,
        [True],
        0,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )
    create_run_path(run_context, ert_config)
    facade = LibresFacade(ert_config)
    with caplog.at_level(logging.ERROR):
        loaded = facade.load_from_forward_model(prior_ensemble, [True], 0)
    expected_loaded, expected_log_message = expected
    assert loaded == expected_loaded
    if expected_log_message:
        assert expected_log_message in "".join(caplog.messages)


def test_load_forward_model_gen_data(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0,1 INPUT_FORMAT:ASCII
        """
    )

    config, prior_ensemble = setup_case(config_text)
    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response_0.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "2", "3"]))
    with open(run_path / "response_1.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["4", "5", "5"]))
    with open(run_path / "response_0.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "0", "1"]))

    facade = LibresFacade(config)
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    prior_ensemble.unify_responses()
    assert list(
        prior_ensemble.load_responses("RESPONSE", (0,))
        .sel(report_step=0, drop=True)
        .to_dataframe()
        .dropna()
        .values.flatten()
    ) == [1.0, 3.0]


def test_single_valued_gen_data_with_active_info_is_loaded(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
        """
    )
    config, prior_ensemble = setup_case(config_text)

    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response_0.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1"]))
    with open(run_path / "response_0.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1"]))

    facade = LibresFacade(config)
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    prior_ensemble.unify_responses()
    assert list(
        prior_ensemble.load_responses("RESPONSE", (0,)).to_dataframe().values.flatten()
    ) == [1.0]


def test_that_all_deactivated_values_are_loaded(setup_case):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response_%d.out REPORT_STEPS:0 INPUT_FORMAT:ASCII
        """
    )
    config, prior_ensemble = setup_case(config_text)

    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response_0.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["-1"]))
    with open(run_path / "response_0.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["0"]))

    facade = LibresFacade(config)
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    prior_ensemble.unify_responses()
    response = (
        prior_ensemble.load_responses("RESPONSE", (0,)).to_dataframe().values.flatten()
    )
    assert np.isnan(response[0])
    assert len(response) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_loading_gen_data_without_restart(storage):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_DATA RESPONSE RESULT_FILE:response.out INPUT_FORMAT:ASCII
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    prior_ensemble = storage.create_ensemble(
        storage.create_experiment(
            responses=ert_config.ensemble_config.response_configuration
        ),
        name="prior",
        ensemble_size=ert_config.model_config.num_realizations,
    )

    run_context = ensemble_context(
        prior_ensemble,
        [True],
        0,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )
    create_run_path(run_context, ert_config)
    run_path = Path("simulations/realization-0/iter-0/")
    with open(run_path / "response.out", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "2", "3"]))
    with open(run_path / "response.out_active", "w", encoding="utf-8") as fout:
        fout.write("\n".join(["1", "0", "1"]))

    facade = LibresFacade.from_config_file("config.ert")
    facade.load_from_forward_model(prior_ensemble, [True], 0)
    prior_ensemble.unify_responses()
    assert list(
        prior_ensemble.load_responses("RESPONSE", (0,))
        .to_dataframe()
        .dropna()
        .values.flatten()
    ) == [1.0, 3.0]


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_that_the_states_are_set_correctly():
    """
    When creating a new ensemble and loading results manually (load_from_forward_model)
    we expect only summary and gen_data to be copied and not parameters.
    """
    facade = LibresFacade.from_config_file("snake_oil.ert")
    storage = open_storage(facade.enspath, mode="w")
    ensemble = storage.get_ensemble_by_name("default_0")
    ensemble_size = facade.get_ensemble_size()
    realizations = np.array([True] * ensemble_size)

    new_ensemble = storage.create_ensemble(
        experiment=ensemble.experiment, ensemble_size=ensemble_size
    )
    facade.load_from_forward_model(new_ensemble, realizations, 0)
    assert not new_ensemble.is_initalized()
    assert new_ensemble.has_data()
