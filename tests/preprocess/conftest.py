import pytest


@pytest.fixture
def data_dir(request):
    return request.path.parent / "data"
