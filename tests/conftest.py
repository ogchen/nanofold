import pytest
from nanofold import mmcif


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


@pytest.fixture
def all_models(data_dir):
    identifiers = mmcif.list_available_mmcif(data_dir)
    assert len(identifiers) == 2
    return {i["id"]: mmcif.load_model(i["id"], i["filepath"]) for i in identifiers}


@pytest.fixture
def model(all_models):
    assert "1A00" in all_models
    return all_models["1A00"]
