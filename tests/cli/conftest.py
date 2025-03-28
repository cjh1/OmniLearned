import tempfile
from pathlib import Path
import requests

import pytest


@pytest.fixture
def test_dataset_name():
    return "toy"


@pytest.fixture
def dataset_base_url():
    return "https://portal.nersc.gov/cfs/m4567/"


@pytest.fixture
def test_dataset_train_url(test_dataset_name, dataset_base_url):
    return f"{dataset_base_url}/{test_dataset_name}/train/val_ttbar_small.h5"


@pytest.fixture
def test_dataset_test_url(test_dataset_name, dataset_base_url):
    return f"{dataset_base_url}/{test_dataset_name}/test/val_ttbar_small.h5"


@pytest.fixture
def test_dataset_val_url(test_dataset_name, dataset_base_url):
    return f"{dataset_base_url}/{test_dataset_name}/val/val_ttbar_small.h5"


@pytest.fixture
def test_dataset_path(
    test_dataset_name,
    test_dataset_train_url,
    test_dataset_test_url,
    test_dataset_val_url,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_url in [
            test_dataset_train_url,
            test_dataset_test_url,
            test_dataset_val_url,
        ]:
            path = Path().joinpath(*file_url.split("/")[-3:])
            file_name = path.name
            download_path = Path(temp_dir, path)
            download_path.parent.mkdir(parents=True, exist_ok=True)
            file_path = Path(temp_dir, file_name)
            with requests.get(file_url, stream=True) as response:
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=16384):  # 16k chunks
                        f.write(chunk)

        yield Path(temp_dir)


@pytest.fixture
def test_dataset_file_paths(
    test_dataset_train_url, test_dataset_test_url, test_dataset_val_url
):
    urls = [test_dataset_train_url, test_dataset_test_url, test_dataset_val_url]

    return [Path().joinpath(*(url.split("/")[-3:-1])) for url in urls]
