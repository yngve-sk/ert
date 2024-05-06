import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture(scope="module", autouse=True)
def set_ert_config(block_storage_path):
    ert_config = ErtConfig.from_file(
        str(block_storage_path / "version-3/poly_example/poly.ert")
    )
    yield local_storage_set_ert_config(ert_config)
    local_storage_set_ert_config(None)


def test_migrate_observations(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-3/poly_example", "poly.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        assert len(list(storage.experiments)) == 1
        experiment = list(storage.experiments)[0]
        assert experiment.observations == {
            k: ert_config.observations.get_dataset(k).squeeze("name", drop=True)
            for k in experiment.observations
        }


def test_migrate_gen_kw_config(setup_case, set_ert_config):
    ert_config = setup_case("block_storage/version-3/poly_example", "poly.ert")
    with open_storage(ert_config.ens_path, "w") as storage:
        experiment = list(storage.experiments)[0]
        assert "template_file_path" not in experiment.parameter_configuration
