import os

import pytest

from instadeep_technical_test.faiss_database import build_and_save_faiss_index


def create_temp_file(directory, file_name, content):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


def test_build_and_save_faiss_index_invalid_extension(tmp_path):
    db_path = tmp_path / "faiss_db"
    db_path.mkdir()
    data_path = "path/to/invalid/files"
    data_file_extension = "csv"  # Assuming csv is an invalid extension
    hf_encoding_model_path = "path/to/hf/model"

    with pytest.raises(ValueError, match="Path path/to/hf/model not found"):
        build_and_save_faiss_index(
            data_path, data_file_extension, hf_encoding_model_path, db_path
        )

    hf_encoding_model_path = "pritamdeka/S-PubMedBert-MS-MARCO"

    with pytest.raises(ValueError, match="Not a valid data type"):
        build_and_save_faiss_index(
            data_path, data_file_extension, hf_encoding_model_path, db_path
        )
