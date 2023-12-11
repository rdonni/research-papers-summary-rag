import os

import click

from instadeep_technical_test.faiss_database import build_and_save_faiss_index
from instadeep_technical_test.streamlit_app import build_streamlit_app


@click.command()
@click.option(
    "--data-path",
    help="path to the folder containing pdf/text files",
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "extracted-text"
    ),
    type=str,
)
@click.option(
    "--data-file-extension",
    help="file extension of data_path's files",
    default="txt",
    type=str,
)
@click.option(
    "--hf-encoding-model-path",
    help="path of the hugging face model for generating embedding",
    default="pritamdeka/S-PubMedBert-MS-MARCO",
    type=str,
)
@click.option(
    "--faiss-index-path", help="path where faiss index will be stored", type=str
)
@click.option(
    "--open-ai-model-name",
    help="name of open ai's llm to use",
    default="gpt-3.5-turbo",
    type=str,
)
@click.option(
    "--open-ai-model-context-size",
    help="context size of the chosen open ai's llm",
    default=4096,
    type=int,
)
@click.option(
    "--num-papers-to-summarize",
    help="number of papers to retrieve and summarize",
    default=3,
    type=int,
)
def main(
    data_path: str,
    data_file_extension: str,
    hf_encoding_model_path: str,
    faiss_index_path: str,
    open_ai_model_name: str,
    open_ai_model_context_size: int,
    num_papers_to_summarize: int,
) -> None:
    faiss_vector_store = build_and_save_faiss_index(
        data_path=data_path,
        data_file_extension=data_file_extension,
        hf_encoding_model_path=hf_encoding_model_path,
        faiss_db_path=faiss_index_path,
    )

    build_streamlit_app(
        faiss_vector_store=faiss_vector_store,
        open_ai_model_name=open_ai_model_name,
        open_ai_model_context_size=open_ai_model_context_size,
        n_papers=num_papers_to_summarize,
    )


if __name__ == "__main__":
    main()
