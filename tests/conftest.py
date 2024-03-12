import pytest
from nanofold.training import mmcif


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


@pytest.fixture
def model(request, data_dir):
    identifiers = mmcif.list_available_mmcif(data_dir)
    matched = [i for i in identifiers if request.param in i]
    assert len(matched) == 1
    return mmcif.load_model(matched[0])
