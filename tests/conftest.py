import pytest
from nanofold import mmcif


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


@pytest.fixture
def all_models(data_dir):
    identifiers = mmcif.list_available_mmcif(data_dir)
    assert len(identifiers) == 2
    models = [mmcif.load_model(i) for i in identifiers]
    return {m.id: m for m in models}


@pytest.fixture
def model(all_models):
    assert "1A00" in all_models
    return all_models["1A00"]
