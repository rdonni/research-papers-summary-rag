from __future__ import annotations

from pathlib import Path

from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class FaissVectorStore:
    def __init__(
        self,
        hf_encoding_model_path: str,
    ):
        """
        Builds a faiss database and save it to faiss_db_path

        :param hf_encoding_model_path: path of hugging face encoder model
        :param faiss_db_path: where to save the DB
        """

        # We define the embedding method
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=hf_encoding_model_path, model_kwargs={"device": "cpu"}
        )

        self.db: FAISS = None

    def build_index(self, data_path: Path | str, data_file_extension: str):
        if data_file_extension == "pdf":
            loader_class = PyPDFLoader
        elif data_file_extension == "txt":
            loader_class = TextLoader
        else:
            raise ValueError("Not a valid data type")

        # We load data from data_path
        data_loader = DirectoryLoader(
            data_path, glob=f"*.{data_file_extension}", loader_cls=loader_class
        )
        documents = data_loader.load()
        self.db = FAISS.from_documents(documents, self.embedding_function)

    def save(self, faiss_db_path: Path):
        """Saves the database
        :param faiss_db_path: Path to the directory where to write the db.
        """
        self.db.save_local(faiss_db_path)

    def load(self, faiss_index_path: Path):
        FAISS.load_local(faiss_index_path, self.embedding_function)


def build_and_save_faiss_index(
    data_path: str,
    data_file_extension: str,
    hf_encoding_model_path: str,
    faiss_db_path: str,
) -> FaissVectorStore:

    store = FaissVectorStore(hf_encoding_model_path)
    store.build_index(data_path, data_file_extension)
    store.save(faiss_db_path)
    return store
