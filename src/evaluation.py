import json
import os
import random
import warnings

import evaluate
import pandas as pd
import ragas
import tiktoken
from datasets import Dataset
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas.metrics import context_relevancy, faithfulness
from ragas.metrics.critique import conciseness

from src.faiss_database import build_and_save_faiss_index
from src.summarize import (
    format_results,
    generate_most_similar_papers_summaries,
)


def evaluation(
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
    """
    This function implements the evaluation pipeline of the RAG
    Metrics computed are ROUGE scores, faithfulness, context_relevancy and conciseness
    :param evaluation_data_path: path to evaluation data folder
    :param evaluation_data_file_extension: file extension in evaluation_data_path
    :param hf_encoding_model_path: path of the hugging face encoder
    :param faiss_index_path: path of the previously created faiss index
    :param nb_queries: number of queries on which to evaluate the RAG system
    :param open_ai_model_name: name of the chosen openAI llm
    :param open_ai_model_context_size: context size of the chosen model
    :param open_ai_api_key: openAI api key
    :param num_papers_to_summarize: number of papers to retrieve for each query
    :param evaluation_results_path: path (containing a json file name) to save evaluation results
    :return:None
    """
    os.environ["OPENAI_API_KEY"] = open_ai_api_key

    # We build a faiss index from files in evaluation_data_path
    vector_store = build_and_save_faiss_index(
        data_path=evaluation_data_path,
        data_file_extension=evaluation_data_file_extension,
        hf_encoding_model_path=hf_encoding_model_path,
        faiss_db_path=faiss_index_path,
    )

    HuggingFaceEmbeddings(
        model_name=hf_encoding_model_path, model_kwargs={"device": "cpu"}
    )

    # We generate some random queries by extracting random 400 characters chunks from evaluation_data_path files
    queries_to_pass_to_rag = extract_queries_to_evaluate_model(
        data_path=evaluation_data_path,
        data_file_extension=evaluation_data_file_extension,
        nb_queries=nb_queries,
        queries_num_characters=400,
    )

    # The context_relevancy metric is a Q/A RAG system metric, so we transform our queries into question so that the
    # metric values make sense
    prompt = "Can you find interesting articles that would help me improve my text below and summarize them for me: "
    queries_with_prompt = [prompt + f"\n{query}" for query in queries_to_pass_to_rag]

    rag_system_results = []
    # For each query generated, we retrieve num_papers_to_summarize relevant papers and generate their summary
    for i, query in enumerate(queries_to_pass_to_rag):
        summaries, context = generate_most_similar_papers_summaries(
            query=query,
            n_papers=num_papers_to_summarize,
            faiss_vector_store=vector_store,
            open_ai_model_name=open_ai_model_name,
            open_ai_model_context_size=open_ai_model_context_size,
            open_ai_api_key=open_ai_api_key,
            return_context=True,
        )
        # summaries is a dict with keys being the titles of retrieved papers and values their respective summary
        # We transform summaries into a readable string with the function format_results
        formatted_summary = format_results(summaries)
        rag_system_results.append(
            {
                "question": queries_with_prompt[i],
                "answer": formatted_summary,
                "contexts": context,
            }
        )

    # We compute rouge metrics on the results of the rag to estimate the quality of the summary
    rouge = evaluate.load("rouge")
    evaluation_results = rouge.compute(
        predictions=[res["answer"] for res in rag_system_results],
        references=["\n".join(res["contexts"]) for res in rag_system_results],
        tokenizer=lambda x: tiktoken.encoding_for_model(open_ai_model_name).encode(x),
    )

    # Some Ragas metrics are computed with a custom llm that belongs to them
    # Depending on the length of the context we pass to it, the metrics computation can fail
    # I need to dig a little deeper into the ragas library to understand how to solve this problem.
    try:
        # We compute the remaining metrics thanks to the library ragas
        rag_system_results = Dataset.from_pandas(pd.DataFrame(data=rag_system_results))
        ragas_results = ragas.evaluate(
            rag_system_results, metrics=[faithfulness, context_relevancy, conciseness]
        )
        # We update the dictionary containing rouge values with new metrics values
        evaluation_results.update(ragas_results)
    except RuntimeError:
        warnings.warn(
            "The context passed to Ragas LLM for metric computation is too large (greater than model context size). Ragas metrics cannot be computed"
        )

    # We save the results to evaluation_results_path
    with open(evaluation_results_path, "w") as outfile:
        json.dump(evaluation_results, outfile)


def extract_queries_to_evaluate_model(
    data_path: str,
    data_file_extension: str,
    nb_queries: int,
    queries_num_characters: int,
) -> list[str]:
    """
    This function loads files from data path and generate nb_queries random queries by extracting random chunks
    :param data_path: path data folder
    :param data_file_extension: file extension in data_path
    :param nb_queries: number of queries to extract
    :param queries_num_characters: size (in characters) of each query
    :return: A list of string queries
    """

    if data_file_extension == "pdf":
        loader_class = PyPDFLoader
    elif data_file_extension == "txt":
        loader_class = TextLoader
    else:
        raise ValueError("Not a valid data_type")

    # We load data from data_path
    data_loader = DirectoryLoader(
        data_path, glob=f"*.{data_file_extension}", loader_cls=loader_class
    )
    documents = data_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=queries_num_characters, chunk_overlap=0
    )
    texts = text_splitter.split_documents(documents)
    random.shuffle(texts)

    query_documents = texts[:nb_queries]
    return [query_documents[i].page_content for i in range(len(query_documents))]
