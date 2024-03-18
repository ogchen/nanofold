import pytest
from nanofold.data_processing.mmcif_processor import list_available_mmcif
from nanofold.data_processing.mmcif_parser import load_model


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"


@pytest.fixture
def model(request, data_dir):
    identifiers = list_available_mmcif(data_dir)
    matched = [i for i in identifiers if request.param in i]
    assert len(matched) == 1
    return load_model(matched[0])
