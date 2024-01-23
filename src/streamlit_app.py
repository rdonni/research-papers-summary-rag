import streamlit as st

from src.api_key_management.extract_api_from_config import (
    get_api_key,
)
from src.faiss_database import FaissVectorStore
from src.summarize import (
    format_results,
    generate_most_similar_papers_summaries,
)


def build_streamlit_app(
    faiss_vector_store: FaissVectorStore,
    open_ai_model_name: str,
    open_ai_model_context_size: int,
    n_papers: int,
) -> None:
    """
    This function build a streamlit app enabling to interact with the RAG system by giving it a few paragraphs as input
    to get a summary of the most related papers from the database.
    :param faiss_index_path: path of the faiss index previously created
    :param faiss_embedding_function: embedding function used during faiss database creation
    :param open_ai_model_name: name of the openAI llm we want to use
    :param open_ai_model_context_size: context size associated to the chosen llm
    :param n_papers: number of papers to retrieve and to summarize for each query
    :return: None
    """

    st.title("🔍 RAG application for most similar papers summarizing")

    openai_api_key = st.sidebar.text_input("OpenAI API Key")

    with st.form("my_form"):
        default_text = """Oral ingestion remains the preferred route for the application of pharmaceuticals, since it does not require a skilled health care professional and allows patients to self-administer drugs conveniently.
DDS for Oral Administration of Biologics. Oral delivery is particularly challenging for biotherapeutics, since these drugs are readily degraded by proteases, nucleases, and other enzymes in the gut, and are much larger than traditional small molecules.
Nordisk is currently developing an orally available long-acting GLP-1 analogue (semaglutide) for the treatment of obesity.
Extended Release DDS for Oral Administration. Traditional oral administration requires frequent dosing as the normal residence time in the GI tract is less than 30 h. """

        text = st.text_area(
            "Enter the research paper extract you would like to improve:", default_text
        )
        submitted = st.form_submit_button("Submit")

        if not openai_api_key.startswith("sk-"):
            openai_api_key = get_api_key()
            if not openai_api_key.startswith("sk-"):
                st.warning(
                    "Please enter your OpenAI API key either in the streamlit field on"
                    " the left or in the config file",
                    icon="⚠",
                )

        if submitted and openai_api_key.startswith("sk-"):
            summaries = generate_most_similar_papers_summaries(
                query=text,
                n_papers=n_papers,
                faiss_vector_store=faiss_vector_store,
                open_ai_model_name=open_ai_model_name,
                open_ai_model_context_size=open_ai_model_context_size,
                open_ai_api_key=openai_api_key,
                return_context=False,
            )

            formatted_response = format_results(summaries)
            st.info(formatted_response)
