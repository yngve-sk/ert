import logging
import re
import shutil
from contextlib import ExitStack

import netCDF4
import numpy as np
import pytest

import ert.storage
import ert.storage.migration.block_fs as bf
from ert.config import ErtConfig, GenKwConfig
from ert.config.responses.summary_config import SummaryConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture
def ensemble(storage):
    return storage.create_experiment().create_ensemble(
        name="default_0", ensemble_size=5
    )


@pytest.fixture(scope="module")
def enspath(block_storage_path):
    return block_storage_path / "snake_oil/storage/snake_oil/ensemble"


@pytest.fixture(scope="module")
def ert_config(block_storage_path):
    return ErtConfig.from_file(str(block_storage_path / "snake_oil/snake_oil.ert"))


@pytest.fixture(scope="module")
def ens_config(ert_config):
    return ert_config.ensemble_config


@pytest.fixture(scope="module", autouse=True)
def set_ert_config(ert_config):
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


@pytest.fixture(scope="module")
def data(block_storage_path):
    return netCDF4.Dataset(block_storage_path / "data_dump/snake_oil.nc")


@pytest.fixture(scope="module")
def forecast(enspath):
    with bf.DataFile(enspath / "default_0/Ensemble/mod_0/FORECAST.data_0") as df:
        yield df


@pytest.fixture(scope="module")
def parameter(enspath):
    with bf.DataFile(enspath / "default_0/Ensemble/mod_0/PARAMETER.data_0") as df:
        yield df


@pytest.fixture(scope="module")
def time_map(enspath):
    return bf._load_timestamps(enspath / "default_0/files/time-map")


def test_migrate_gen_kw(data, parameter, ens_config, tmp_path):
    group_root = "/REAL_0/GEN_KW"
    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            parameters=[
                GenKwConfig(
                    name="SNAKE_OIL_PARAM",
                    forward_init=False,
                    template_file="",
                    transfer_function_definitions=[],
                    output_file="kw.txt",
                    update=True,
                )
            ]
        )
        ensemble = experiment.create_ensemble(name="default_0", ensemble_size=5)
        bf._migrate_gen_kw(ensemble, parameter, ens_config)

        for param in ens_config.parameters:
            expect_names = list(data[f"{group_root}/{param}"]["name"])
            expect_array = np.array(data[f"{group_root}/{param}"]["standard_normal"])
            actual = ensemble.load_parameters(param, 0)

            assert expect_names == list(actual["names"]), param
            assert (expect_array == actual).all(), param


def test_migrate_summary(data, forecast, time_map, tmp_path):
    group = "/REAL_0/SUMMARY"
    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[
                SummaryConfig(name="summary", input_file="some_file", keys=["some_key"])
            ]
        )
        ensemble = experiment.create_ensemble(name="default_0", ensemble_size=5)

        bf._migrate_summary(ensemble, forecast, time_map)

        expected_keys = set(data[group].variables) - set(data[group].dimensions)
        assert set(ensemble.get_summary_keyset()) == expected_keys

        for key in ensemble.get_summary_keyset():
            expect = np.array(data[group][key])[1:]  # Skip first report_step
            actual = (
                ensemble.load_responses("summary", (0,))
                .sel(name=key)["values"]
                .data.flatten()
            )
            assert list(expect) == list(actual), key


def test_migrate_gen_data(data, forecast, tmp_path):
    group = "/REAL_0/GEN_DATA"
    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            responses=[
                SummaryConfig(name=name, input_file="some_file", keys=["some_key"])
                for name in (
                    "SNAKE_OIL_WPR_DIFF",
                    "SNAKE_OIL_OPR_DIFF",
                    "SNAKE_OIL_GPR_DIFF",
                )
            ]
        )
        ensemble = experiment.create_ensemble(name="default_0", ensemble_size=5)
        bf._migrate_gen_data(ensemble, forecast)

        for key in set(data[group].variables) - set(data[group].dimensions):
            expect = np.array(data[group][key]).flatten()
            actual = ensemble.load_responses(key, (0,))["values"].data.flatten()
            assert list(expect) == list(actual), key


@pytest.mark.parametrize("name,iter", [("default_3", 3), ("foobar", 0)])
def test_migrate_case(data, storage, tmp_path, enspath, ens_config, name, iter):
    shutil.copytree(enspath, tmp_path, dirs_exist_ok=True)
    (tmp_path / "default_0").rename(tmp_path / name)
    with ExitStack() as stack:
        bf.migrate_case(storage, tmp_path / name, stack)

    ensemble = storage.get_ensemble_by_name(name)
    assert ensemble.iteration == iter
    for real_key, var in data.groups.items():
        index = int(re.match(r"REAL_(\d+)", real_key)[1])

        # Sanity check: Test data only contains GEN_KW, GEN_DATA and SUMMARY
        assert set(var.groups) == {"GEN_KW", "GEN_DATA", "SUMMARY"}

        # Compare SUMMARYs
        for key in ensemble.get_summary_keyset():
            expect = np.array(var["SUMMARY"][key])[1:]  # Skip first report_step
            actual = (
                ensemble.load_responses("summary", (index,))
                .sel(name=key)["values"]
                .data.flatten()
            )
            assert list(expect) == list(actual), key

        # Compare GEN_KWs
        for param in ens_config.parameters:
            expect_names = list(var[f"GEN_KW/{param}"]["name"])
            expect_array = np.array(var[f"GEN_KW/{param}"]["standard_normal"])
            actual = ensemble.load_parameters(param, index)

            assert expect_names == list(actual["names"]), param
            assert (expect_array == actual).all(), param

        # Compare GEN_DATAs
        for key in set(var["GEN_DATA"].variables) - set(var["GEN_DATA"].dimensions):
            expect = np.array(var["GEN_DATA"][key]).flatten()
            actual = ensemble.load_responses(key, (index,))["values"].data.flatten()
            assert list(expect) == list(actual), key


def test_migration_failure(storage, enspath, ens_config, caplog, monkeypatch):
    """Run migration but fail due to missing config data. Expected behaviour is
    for the error to be logged but no exception be propagated.

    """
    monkeypatch.setattr(ens_config, "parameter_configs", {})
    monkeypatch.setattr(ert.storage, "open_storage", lambda: storage)

    # Sanity check: no ensembles are created before migration
    assert list(storage.ensembles) == []

    with caplog.at_level(logging.WARNING, logger="ert.storage.migration.block_fs"):
        bf._migrate_case_ignoring_exceptions(storage, enspath / "default_0")

    # No ensembles were created due to failure
    assert list(storage.ensembles) == []

    # Warnings are in caplog
    assert len(caplog.records) == 1
    assert caplog.records[0].message == (
        "Exception occurred during migration of BlockFs case 'default_0': "
        "'The key:SNAKE_OIL_PARAM is not in the ensemble configuration'"
    )


@pytest.mark.parametrize("should_fail", [False, True])
def test_full_migration_logging(
    tmp_path, enspath, caplog, monkeypatch, should_fail, ens_config
):
    if should_fail:
        monkeypatch.setattr(ens_config, "parameter_configs", {})

    caplog.set_level(logging.INFO)
    shutil.copytree(enspath, tmp_path / "storage")

    with ert.storage.open_storage(tmp_path / "storage", mode="w"):
        pass

    msgs = [r.message for r in caplog.records]
    assert "Outdated storage detected at" in msgs.pop(0)
    assert "Backing up BlockFs storage" in msgs.pop(0)
    assert msgs.pop(0) == "Migrating case 'default'"
    assert msgs.pop(0) == "Migrating case 'default_0'"

    if should_fail:
        assert msgs.pop(0) == (
            "Exception occurred during migration of BlockFs case 'default_0': "
            "'The key:SNAKE_OIL_PARAM is not in the ensemble configuration'"
        )
        assert msgs.pop(0) == "Migration from BlockFs completed with 1 failure(s)"
    else:
        assert msgs.pop(0) == "Migration from BlockFs completed with 0 failure(s)"

    assert "Note: ERT 4 and lower is not compatible" in msgs.pop(0)
