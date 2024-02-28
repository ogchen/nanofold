import pytest
from nanofold import mmcif


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


@pytest.fixture
def model(data_dir):
    identifiers = mmcif.list_available_mmcif(data_dir)
    assert len(identifiers) == 1
    assert identifiers[0]["id"] == "1A00"
    return mmcif.load_model(identifiers[0]["id"], identifiers[0]["filepath"])
