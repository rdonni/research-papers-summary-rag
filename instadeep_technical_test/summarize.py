from __future__ import annotations

import logging
import os

import tiktoken
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from instadeep_technical_test.faiss_database import FaissVectorStore

LOGGER = logging.getLogger(__name__)


def generate_most_similar_papers_summaries(
    query: str,
    n_papers: int,
    faiss_vector_store: FaissVectorStore,
    open_ai_model_name: str,
    open_ai_model_context_size,
    open_ai_api_key: str,
    return_context: bool,
) -> dict[str, str] | tuple[dict[str, str], list[str]]:
    """
    This function, provided a query, retrieves the n_papers most similar papers based on a previously created
    faiss index and generate a summary for each of those
    :param query: a string representing a paper or a paper extract we want to improve
    :param n_papers: number of articles similar to the query we want to retrieve (and to summarize)
    :param faiss_vector_store: Instance of FaissVectorStore defined in faiss_database.py
    :param open_ai_model_name: name of the openAI llm we want to use
    :param open_ai_model_context_size: context size of the chosen llm
    :param open_ai_api_key: openAI api key
    :param return_context: boolean indicating if we return the context that enabled the model to generate the summaries
    :return: A dictionary with keys being the titles of retrieved papers and values their respective summary
     + the context (a list of the relevant papers full text) if return_context=True
    """

    # Build ChatOpenAI llm instance
    llm = build_open_ai_llm(open_ai_model_name, open_api_key=open_ai_api_key)

    # Retrieve the n_papers papers most similar to query
    retriever = faiss_vector_store.db.as_retriever(search_kwargs={"k": n_papers})
    relevant_papers = retriever.get_relevant_documents(query)

    # Extract name and number of tokens of relevant_papers
    papers_names = [
        extract_file_name_from_path(relevant_papers[i].metadata["source"])
        for i in range(len(relevant_papers))
    ]
    relevant_papers_num_tokens = [
        num_tokens_from_string(relevant_papers[i].page_content, open_ai_model_name)
        for i in range(len(relevant_papers))
    ]

    # Browse relevant papers to compute their summary using llm built previously
    LOGGER.info("Computing summaries of the fetched documents")
    summaries = {}
    for i in range(len(relevant_papers)):
        if relevant_papers_num_tokens[i] < open_ai_model_context_size:
            # If the document is shorter than open_ai_model_context_size we can use a "stuff" chain
            # (i.e. the document is entirely passed as a model input)
            chain = build_summarize_chain(llm, chain_type="stuff")
            document = [Document(page_content=relevant_papers[i].page_content)]
        else:
            LOGGER.info(
                f"Using map-reduce to deal with large context of size {relevant_papers_num_tokens[i]}"
            )
            # If the document is longer than open_ai_model_context_size we need to use a "map_reduce" or "refine" chain
            # (i.e. the global summary is built as a composite of document chunks summaries)
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                model_name=open_ai_model_name
            )
            splitted_text = text_splitter.split_text(relevant_papers[i].page_content)
            document = [Document(page_content=t) for t in splitted_text]
            chain = build_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(document)
        summaries.update({papers_names[i]: summary})

    if return_context:
        return summaries, [
            relevant_papers[i].page_content for i in range(len(relevant_papers))
        ]
    return summaries


def build_open_ai_llm(model_name: str, open_api_key: str) -> ChatOpenAI | None:
    """
    This function builds a ChatOpenAI instance if open_api_key is a valid api key
    :param model_name: openAI llm name
    :param open_api_key: openAI api key
    :return: a ChatOpenAI instance or None if the api key is invalid
    """
    try:
        # We use a low temperature value as we don't want the model to be imaginative but to stick to the data from
        # database
        llm = ChatOpenAI(
            temperature=0.1, openai_api_key=open_api_key, model_name=model_name
        )
        return llm
    except Exception as e:
        raise ValueError("Not a valid open ai api key") from e


def build_summarize_chain(model: ChatOpenAI, chain_type: str):
    """
    Build a langchain summarize chain
    :param model: openAI llm name
    :param chain_type: Langchain summarize_chain chain type (to choose among stuff and map_reduce)
    :return:
    """
    prompt_template = """Write a short summary of the following medical research paper:
            {text}
            CONCISE SUMMARY:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if chain_type == "stuff":
        chain = load_summarize_chain(model, chain_type=chain_type, prompt=prompt)
    elif chain_type == "map_reduce":
        chain = load_summarize_chain(
            model,
            chain_type=chain_type,
            map_prompt=prompt,
            combine_prompt=prompt,
            verbose=False,
        )
    else:
        raise ValueError("Not a valid chain type")
    return chain


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Count the number of tokens in a given string using the tiktoken tokenizer
    :param string:
    :param encoding_name: openAI model name
    :return: number of tokens in string
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    return len(encoding.encode(string))


def extract_file_name_from_path(path: str) -> str:
    """
    Extract the name of a file without file extension from a path
    :param path:
    :return: string
    """
    file_name = os.path.basename(path)
    return os.path.splitext(file_name)[0]


def format_results(papers_summaries_dict: dict[str, str]) -> str:
    """
    Function which reformat generate_most_similar_papers_summaries dictionary output
    :param papers_summaries_dict: output of generate_most_similar_papers_summaries
    :return: a string containing all information from papers_summaries_dict
    """
    introduction = "Here are the most relevant papers I've found to help you:\n"
    results = ""

    for title, summary in papers_summaries_dict.items():
        paper_info = f"\n- Title: {title}\n\nSummary: {summary}\n"
        results += paper_info

    return introduction + results
