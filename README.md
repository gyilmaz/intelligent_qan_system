# Intelligent Document QA System

## Project Description

This project implements an Intelligent Document Question Answering system using the Retrieval Augmented Generation (RAG) pattern. The system processes a PDF document (like a school handbook), creates a searchable knowledge base from its content, and uses a Language Model (LLM) to answer user questions based *only* on the information found in the document.

The project is structured as a pipeline using Hydra for configuration management and can be extended for different document types, embedding models, LLMs, and deployment targets.

## Features

* **PDF Processing:** Extracts text content from PDF documents and splits it into manageable chunks.
* **Vector Database Indexing:** Creates embeddings (numerical representations) for text chunks using a Sentence Transformer model and stores them in a persistent ChromaDB vector database.
* **Retrieval Augmented Generation (RAG):** Answers user questions by:
    * Embedding the user query.
    * Searching the vector database for relevant text chunks.
    * Providing the retrieved chunks as context to a generator LLM.
    * Generating an answer from the LLM based on the provided context.
* **Configurable Pipeline:** Uses Hydra to manage configuration for data paths, model names, database locations, and pipeline steps.
* **FastAPI:** Provides a web API endpoint to receive user questions and return answers.

## Project Structure
intelligent_qan_system/
:start_line:23
-------
├── .env                        # Environment variables
├── .gitignore                  # Specifies intentionally untracked files that Git should ignore
├── conf/
│   ├── config.yaml             # Main configuration file
│   ├── data/
│   │   └── default.yaml        # Data processing configuration
│   ├── deployment/
│   │   └── default.yaml        # Deployment configuration (e.g., AWS settings)
│   ├── model/
│   │   └── default.yaml        # Model configuration (embedding, generator LLM, vector DB)
│   └── api/
:start_line:39
-------
│       └── default.yaml        # API configuration (host, port)
├── data/
│   ├── raw/                    # Directory for raw input data (e.g., your PDF)
│   ├── processed/              # Directory for processed data (text chunks)
│   └── vector_db/              # Directory for the ChromaDB vector database
│       ├── chroma.sqlite3       # ChromaDB database file
│       └── [UUID]/            # ChromaDB data files
├── outputs/                    # Directory for experiment outputs and logs
├── src/
│   ├── __init__.py             # Makes src a Python package
│   ├── data_processing.py      # Logic for reading and chunking documents
│   ├── vector_db.py            # Logic for creating and interacting with the vector database
│   ├── rag_system.py           # Core RAG query logic (retrieval + generation)
│   └── api.py                  # FastAPI application and endpoint definition
│   ├── data_processing.py      # Logic for reading and chunking documents
│   ├── vector_db.py            # Logic for creating and interacting with the vector database
│   ├── rag_system.py           # Core RAG query logic (retrieval + generation)
│   └── api.py                  # FastAPI application and endpoint definition
├── run.py                      # Main script to run pipeline steps using Hydra
├── requirements.txt            # Project dependencies
└── README.md                   # This file

## Prerequisites

* Python 3.8+
* Git (for cloning the repository)
* `pip` (Python package installer)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd intelligent_qan_system
    ```
2.  **Create a Python Virtual Environment:**
    ```powershell
    python -m venv venv
    ```
3.  **Activate the Virtual Environment:**
    ```powershell
    .\venv\Scripts\Activate.ps1 # On Windows PowerShell
    # source venv/bin/activate   # On macOS/Linux Bash
    ```
4.  **Install Dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```
5.  **Place Your Document:** Put your PDF handbook file (e.g., `2024-2025_HPS_Student_and_Parent_Handbook.pdf`) inside the `data/raw/` directory.

## Configuration

Configuration is managed in the `conf/` directory using Hydra.

* `conf/config.yaml`: The main entry point, defining which default configurations to load (`data`, `model`, `deployment`, `api`) and the overall project name.
* `conf/data/default.yaml`: Specifies the input PDF path, output processed data path, and chunking parameters. **Update `raw_data_path`** to point to your specific PDF file name.
* `conf/model/default.yaml`: Defines the embedding model name, generator LLM name, and the path for the vector database.
* `conf/deployment/default.yaml`: Placeholder for future deployment settings (e.g., AWS).
* `conf/api/default.yaml`: Defines the host and port for the FastAPI application.

Review and adjust the values in these files as needed for your setup.

## Running the Pipeline

The project uses `run.py` with Hydra to execute different pipeline steps. Ensure your virtual environment is activated.

* **Process Data:** Read the PDF and create text chunks.
    ```powershell
    python run.py pipeline=process_data
    ```
    This will save processed chunks to `data/processed/processed_handbook_chunks.json`.

* **Build Vector Index:** Generate embeddings for chunks and build the ChromaDB index.
    ```powershell
    python run.py pipeline=build_index
    ```
    This will create the vector database files in `data/vector_db/`. This step downloads the embedding model the first time.

* **Run API:** Start the FastAPI application.
    ```powershell
    python run.py pipeline=run_api
    ```
    This will start the Uvicorn server, typically on `http://0.0.0.0:8000`. The API startup loads models and connects to the database via the `lifespan` event handler in `src/api.py`. This step downloads the generator LLM the first time.

## Testing the API

Once the API is running (`python run.py pipeline=run_api` in one terminal), you can send POST requests to the `/answer` endpoint from another terminal or a web browser.

* **Using PowerShell `Invoke-WebRequest`:**
    ```powershell
    Invoke-WebRequest -Uri [http://127.0.0.1:8000/answer](http://127.0.0.1:8000/answer) -Method POST -Headers @{"Content-Type" = "application/json"} -Body '{"question": "What is the school\'s policy on student conduct?"}'
    ```

* **Using Swagger UI:**
    Open your web browser and navigate to `http://127.0.0.1:8000/docs`. You can use the interactive documentation to test the `/answer` endpoint.

## Current State

The core RAG pipeline (Process Data -> Build Index -> Run API -> Query) is functional.

* Data processing and index building are working.
* The FastAPI application starts and loads resources on startup.
* The API endpoint receives questions, retrieves relevant chunks from ChromaDB, and calls the generator LLM.
* The generator LLM used in the current configuration (`distilbert/distilgpt2`) is a very small model and produces low-quality, often nonsensical answers. The ChromaDB query crash issue on Windows seems to be resolved (potentially due to the `hnsw:batch_size` workaround and/or Python version).

## Future Work

* **Improve Generator LLM:** Replace the small generator LLM with a more capable model (e.g., larger open-source models or API-based models like Gemini, GPT-4, Claude) for better answer quality.
* **Implement Streamlit UI:** Build a user interface to interact with the API more easily.
* **Deployment:** Containerize the application (e.g., using Docker) and deploy it to a cloud platform (e.g., AWS ECS).
* **Error Handling & Logging:** Enhance error handling and add more robust logging.
* **Add More Document Types:** Extend data processing to handle other document formats (e.g., `.docx`, `.txt`).
* **Advanced RAG Techniques:** Explore techniques like re-ranking retrieved documents, query expansion, or different prompting strategies.

---
