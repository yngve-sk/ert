import pathlib
from datetime import datetime

import numpy as np
import pytest

from ert.config import ErtConfig
from ert.data import MeasuredData
from ert.libres_facade import LibresFacade
from ert.storage import open_storage


@pytest.fixture()
def facade_snake_oil(snake_oil_case_storage):
    yield LibresFacade(snake_oil_case_storage)


@pytest.fixture
def default_ensemble(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ens_path) as storage:
        yield storage.get_ensemble_by_name("default_0")


@pytest.fixture
def create_measured_data(snake_oil_case_storage, snake_oil_default_storage):
    def func(*args, **kwargs):
        return MeasuredData(
            snake_oil_default_storage,
            *args,
            **kwargs,
        )

    return func


def test_history_obs(create_measured_data):
    fopr = create_measured_data(["FOPR"])
    fopr.remove_inactive_observations()

    assert len(fopr.data.columns) == 200


def test_summary_obs(create_measured_data):
    summary_obs = create_measured_data(["WOPR_OP1_72"])
    summary_obs.remove_inactive_observations()
    # Only one observation, we check the key_index is what we expect:
    assert (
        summary_obs.data.columns.get_level_values("key_index").values[0] == "2011-12-21"
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.usefixtures("copy_snake_oil_case")
@pytest.mark.parametrize("formatted_date", ["2015-06-23", "23/06/2015"])
def test_summary_obs_last_entry(formatted_date):
    obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
    with obs_file.open(mode="w") as fin:
        fin.write(
            "\n"
            "SUMMARY_OBSERVATION LAST_DATE\n"
            "{\n"
            "   VALUE   = 10;\n"
            "   ERROR   = 0.1;\n"
            f"   DATE    = {formatted_date};\n"
            "   KEY     = FOPR;\n"
            "};\n"
        )
    observation = ErtConfig.from_file("snake_oil.ert").observations
    assert list(observation["LAST_DATE"].observations) == [datetime(2015, 6, 23, 0, 0)]


def test_gen_obs(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1"])
    df.remove_inactive_observations()

    assert sorted(df.data.columns.get_level_values("key_index").values) == [
        "[1200, 199]",
        "[1800, 199]",
        "[400, 199]",
        "[800, 199]",
    ]


def test_gen_obs_and_summary(create_measured_data):
    df = create_measured_data(["WPR_DIFF_1", "WOPR_OP1_9"])
    df.remove_inactive_observations()

    assert sorted(df.data.columns.get_level_values(0).to_list()) == sorted(
        [
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WPR_DIFF_1",
            "WOPR_OP1_9",
        ]
    )


def test_gen_obs_and_summary_index_range(create_measured_data):
    df = create_measured_data(
        ["WPR_DIFF_1", "FOPR"],
        [["[800, 199]"], [datetime(2010, 4, 20).strftime("%Y-%m-%d")]],
    )
    df.remove_inactive_observations()

    assert df.data.columns.get_level_values(0).to_list() == ["WPR_DIFF_1", "FOPR"]
    assert df.data.loc["OBS"].values == pytest.approx([0.1, 0.23281], abs=0.00001)
    assert df.data.loc["STD"].values == pytest.approx([0.2, 0.1])


@pytest.mark.parametrize(
    "obs_key, expected_msg",
    [
        ("FOPR", r"No observation: FOPR in ensemble"),
        ("WPR_DIFF_1", "No observation: WPR_DIFF_1 in ensemble"),
    ],
)
def test_no_storage(obs_key, expected_msg, storage):
    ensemble = storage.create_experiment().create_ensemble(
        name="empty", ensemble_size=10
    )

    with pytest.raises(
        KeyError,
        match=expected_msg,
    ):
        MeasuredData(ensemble, [obs_key])


def create_summary_observation():
    observations = ""
    values = np.random.uniform(0, 1.5, 200)
    errors = values * 0.1
    for restart, (value, error) in enumerate(zip(values, errors)):
        observations += f"""
    \nSUMMARY_OBSERVATION FOPR_{restart + 1}
{{
    VALUE   = {value};
    ERROR   = {error};
    RESTART = {restart + 1};
    KEY     = FOPR;
}};
    """
    return observations


def create_general_observation():
    observations = ""
    index_list = np.array(range(2000))
    index_list = [index_list[i : i + 4] for i in range(0, len(index_list), 4)]
    for nr, (i1, i2, i3, i4) in enumerate(index_list):
        observations += f"""
    \nGENERAL_OBSERVATION CUSTOM_DIFF_{nr}
{{
   DATA       = SNAKE_OIL_WPR_DIFF;
   INDEX_LIST = {i1},{i2},{i3},{i4};
   RESTART    = 199;
   OBS_FILE   = wpr_diff_obs.txt;
}};
    """
    return observations


def test_all_measured_snapshot(snapshot, facade_snake_oil, create_measured_data):
    """
    While there is no guarantee that this snapshot is 100% correct, it does represent
    the current state of loading from storage for the snake_oil case.
    """
    measured_data = create_measured_data()
    snapshot.assert_match(measured_data.data.to_csv(), "snake_oil_measured_output.csv")
