import os

import click

from src.api_key_management.extract_api_from_config import (
    get_api_key,
)
from src.evaluation import evaluation


@click.command()
@click.option(
    "--evaluation-data-path",
    help="path to the folder containing pdf/text files",
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "extracted-text"
    ),
    type=str,
)
@click.option(
    "--evaluation-data-file-extension",
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
    "--nb-queries",
    help="number of queries to generate for evaluation",
    default=2,
    type=int,
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
    "--open-ai-api-key", help="open ai api key", default=get_api_key(), type=str
)
@click.option(
    "--num-papers-to-summarize",
    help="number of papers to retrieve and summarize",
    default=1,
    type=int,
)
@click.option(
    "--evaluation-results-path",
    help="path (with json file name) where evaluation results will be stored",
    type=str,
)
def evaluate(
    evaluation_data_path: str,
    evaluation_data_file_extension: str,
    hf_encoding_model_path: str,
    faiss_index_path: str,
    nb_queries: int,
    open_ai_model_name: str,
    open_ai_model_context_size: int,
    open_ai_api_key: str,
    num_papers_to_summarize: int,
    evaluation_results_path: str,
) -> None:
    evaluation(
        evaluation_data_path=evaluation_data_path,
        evaluation_data_file_extension=evaluation_data_file_extension,
        hf_encoding_model_path=hf_encoding_model_path,
        faiss_index_path=faiss_index_path,
        nb_queries=nb_queries,
        open_ai_model_name=open_ai_model_name,
        open_ai_model_context_size=open_ai_model_context_size,
        open_ai_api_key=open_ai_api_key,
        num_papers_to_summarize=num_papers_to_summarize,
        evaluation_results_path=evaluation_results_path,
    )


if __name__ == "__main__":
    evaluate()
