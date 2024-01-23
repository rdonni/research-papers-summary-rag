# 🔍 Retrieval Augmented Generation for most similar document summarization

## 📢  Presentation
This repository presents a RAG architecture for summarizing research papers retrieved by similarity search against a user query.

## ⚙️ Description of main components
The main components of the system are described below:

- **Vector DataBase** : For the database I decided to use FAISS. It's a database optimised for similarity search, and there's an api to use it directly in langchain.The FAISS library also offers techniques such as product quantization for approximate nearest neighbour search. However, this functionality does not seem to be directly available in the langchain api. But the exact nearest neighbour search is sufficient for this project, given the small number of documents we need to embed.


- **Embeddings**: To create the embeddings I used a sentence bert available on hugging face, trained on a dataset of medical research papers (available [here](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO)). The inference is quite fast even on CPU and given that the papers we need to embed are medical research papers, this makes it a good candidate.


- **Generative model** :  For the LLM I had initially chosen a quantized version of llama2 - 7B but the inference remained very long. I therefore decided to use the openAI API to reduce the inference time (more specifically the gpt-3.5-turbo model) which is easily integrated into langchain. For our summarisation use case, we don't want the model to be imaginative, so we chose a low temperature value equal to 0.1.


### Additional thoughts
The documents handled here are relatively short. They fit within the context size of the model (4096 tokens) or two context sizes for longer papers. When the document is longer than the context size, we can use langchain's map_reduce and refine chain types to summarise the document without having to make too many calls to the API.

If the documents were much longer, and it became too expensive to summarise them, we could divide them into several chunks and use a K-NN algorithm to cluster them. We could then retrieve the extracts closest to each centroid (i.e. the extracts most representative of the entire paper) and send only these extracts to the LLM for summarisation.  This is not necessarily necessary here.

## 🛠️ Software engineering practices
During this project I used the following features:

- **poetry** for managing libraries

- **pytest** for testing and the pytest-cov plugin for measuring test coverage

- the **pre-commit** framework to manage the pre-commit pipeline (containing the **black** and **ruff** linter)

- the **click** library for creating command line interfaces

## 📂 Project structure
Here is the structure of the source folder for my project:

[instadeep_technical_test](src) :
- [api_key_management](src/api_key_management)
  - [__init__.py](src/api_key_management/__init__.py)
  - [config.ini](src/api_key_management/config.ini): file where to write the openAI API key
  - [extract_api_from_config.py](src/api_key_management/extract_api_from_config.py): This file contains a function to automatically extract the API key from config.ini
- [evaluation.py](src/evaluation.py): This file contains the implementation of the evaluation pipeline
- [faiss_database.py](src/faiss_database.py): This file permits the faiss index creation and saving
- [streamlit_app.py](src/streamlit_app.py): This file builds the streamlit_app enabling to interact with the system
- [summarize.py](src/summarize.py): This file contains all the generative part (instanciation of the llm model + generation of summaries )
- [cli](src/cli): CLIs for the project

The command line interface for launching the project on the command line can be found in the cli folder.


## 📊 Evaluation process
First of all, we must generate potential user queries to be able to pass them to our RAG. For this I extracted random chunks of research papers from the data files provided.

Any RAG implementation has two aspects: Generation and Retrieval. For a complete evaluation of the system we need to assess both aspects. We don't have a ground truth (neither for the retrieving nor for the generation), so this limits our choice of metrics.
Several metrics that I use are implemented via the ragas library, which offers a wide number of LLM-based metrics.

### 1) Retrieval
- **Context relevancy** : This metric was originally designed for question answering tasks. It measures the relevance of the retrieved context, provided a certain user query. Given that we are not exactly in a QA context, this metric can be complicated to interpret, but it can still be interesting to estimate.

### 2) Generation
- **ROUGE** : ROUGE metrics are often used to quantify the quality of a summary based on the original document (by counting the number of n-gram overlaps). I used the Hugging Face evaluation library to calculate this metric.


- **Faithfullness** : Looking at the context passed to a LLM and its answer, Faithfullness (from ragas library) quantifies the hallucinations and the accuracy of the LLM answer.


- **Conciseness** : Conciseness is an LLM-based evaluation criteria available in ragas library that measures to what extent the submission is concise and to the point which is quite relevant for a summarization task.

As most of these metrics are based on LLMs, the evaluation pipeline can take a long time, even on a small number of examples.
Also, if we choose to retrieve too many papers for each of the queries in the evaluation dataset, the context passed to Ragas' LLMs sometimes becomes longer than its context size, which creates an error.
I'll have to dig a little deeper to figure out how to fix this.

## ▶️ To run the code
First of all, to be able to use the OpenAI API, you must add your personal API key in the [config.ini](src/api_key_management/config.ini) file.

Once this is done, you must install the libraries used in the project via poetry:
```shell
poetry install
```

Install in dev mode:
```shell
poetry install --with dev
```

Run tests:
```shell
poetry run pytest tests
```


To launch the streamlit application, from the project root:
```shell
poetry run -- streamlit run src/cli/main.py -- --faiss-index-path /your/path/to/faiss/index
```
A streamlit page will then open, simply press the submit button and wait a few tens of seconds to obtain the response from the model.


To launch the evaluation pipeline, from the project root:
```shell
poetry run python3 src/cli/evaluation.py --faiss-index-path /your/path/to/faiss/index --evaluation-results-path /your/path/to/results.json
```
