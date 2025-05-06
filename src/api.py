# src/api.py
import os
print("Config file exists?", os.path.exists("conf/config.yaml"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from omegaconf import DictConfig, OmegaConf
import uvicorn
# Import sys to potentially exit on critical errors
import sys

# Import your RAG function and necessary components
from src.rag_system import get_rag_answer
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Define request and response models using Pydantic
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# --- Global variables to hold models, DB connection, and config ---
embedding_model = None
generator_llm_tokenizer = None
generator_llm_model = None
vector_db_collection = None
app_config: DictConfig = None # This will be set by initialize_api_config_and_models

def initialize_api_config_and_models(hydra_cfg: DictConfig):
    """
    Initializes the global app_config and loads models and DB.
    Called by run.py with the fully resolved Hydra configuration.
    """
    global embedding_model, generator_llm_tokenizer, generator_llm_model, vector_db_collection, app_config
    
    print("API init: Setting global app_config from Hydra.")
    app_config = hydra_cfg

    if not app_config:
        print("API init ERROR: Received None for hydra_cfg. Cannot initialize.")
        return

    print(f"API init debug: app_config is None? {app_config is None}")
    if app_config:
        print(f"API init debug: 'project_name' in app_config? {'project_name' in app_config and app_config.project_name is not None}")
        print(f"API init debug: 'data' in app_config? {'data' in app_config and app_config.data is not None}")
        print(f"API init debug: 'model' in app_config? {'model' in app_config and app_config.model is not None}")
        print(f"API init debug: 'api' in app_config? {'api' in app_config and app_config.api is not None}")

    # --- Load Models and DB ---
    if app_config and all(key in app_config for key in ['model', 'data', 'project_name']) and \
       hasattr(app_config, 'model') and app_config.model and \
       hasattr(app_config.model, 'embedding_model_name') and \
       hasattr(app_config.model, 'generator_llm_name') and \
       hasattr(app_config.model, 'vector_db_path') and \
       hasattr(app_config, 'project_name'):
        print("API init: Loading models and connecting to DB...")
        try:
            embedding_model = SentenceTransformer(app_config.model.embedding_model_name, device='cpu')
            print("API init: Embedding model loaded.")

            generator_llm_tokenizer = AutoTokenizer.from_pretrained(app_config.model.generator_llm_name)
            generator_llm_model = AutoModelForCausalLM.from_pretrained(app_config.model.generator_llm_name)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            generator_llm_model.to(device)
            print(f"API init: Generator LLM and tokenizer loaded. Using device: {device}")

            client = chromadb.PersistentClient(path=app_config.model.vector_db_path)
            collection_name = app_config.project_name.lower().replace(" ", "_") + "_collection"
            vector_db_collection = client.get_or_create_collection(name=collection_name) # Changed to get_or_create
            print(f"API init: Ensured ChromaDB collection exists: {collection_name}")

            print("API init: Models and DB connected successfully!")

        except Exception as e:
            print(f"API init: Error loading models or connecting to DB: {e}")
            embedding_model = None
            generator_llm_tokenizer = None
            generator_llm_model = None
            vector_db_collection = None
    else:
        print("API init: Skipping model/DB loading because critical configuration keys/sections are missing or incomplete.")

# --- Lifespan Event Handler ---
async def lifespan(app_instance: FastAPI): # Renamed 'app' to 'app_instance'
    """
    Handles application startup and shutdown events.
    Model and config loading is now handled by initialize_api_config_and_models.
    """
    global app_config # Ensure app_config is accessible if needed for other lifespan tasks
    print("API lifespan: Entered.")
    
    # If you have other startup tasks that depend on app_config, check it here.
    if not app_config:
        print("API lifespan WARNING: app_config is not set. This might be an issue if other lifespan tasks depend on it.")

    yield # FastAPI proceeds

    print("API lifespan: Shutting down...")
    # Add cleanup code here if necessary
    print("API lifespan: Shutdown complete.")


# --- FastAPI App Instance ---
# Pass the lifespan function to the FastAPI constructor
app = FastAPI(
    title="Intelligent Document QA System API",
    description="API for querying document knowledge base",
    version="0.1.0",
    lifespan=lifespan # <--- Pass the lifespan function here
)


# --- API Endpoint ---
@app.post("/answer", response_model=AnswerResponse)
async def answer_question_endpoint(request: QuestionRequest):
    """
    Receives a question and returns an answer based on the handbook.
    """
    # Check if models and DB were loaded successfully on startup
    if embedding_model is None or generator_llm_model is None or vector_db_collection is None or app_config is None:
        print("API /answer ERROR: Service not ready. Models, DB, or app_config not loaded.")
        raise HTTPException(status_code=503, detail="Service not ready: Models or database not loaded correctly on startup.")

    question = request.question
    print(f"API /answer: Received question: '{question}'")

    try:
        answer = get_rag_answer(
            app_config,
            question,
        )
        print(f"API /answer: Generated answer: '{answer}'")
        return AnswerResponse(answer=answer)
    except Exception as e:
        print(f"API /answer: CRITICAL ERROR during get_rag_answer or response generation: {e}")
        # Log the stack trace for more details
        import traceback
        print(traceback.format_exc())
        # Return a 500 error to the client
        raise HTTPException(status_code=500, detail=f"Internal server error while processing question: {str(e)}")

# --- Helper function to run the API (called by run.py) ---
# This function should be at the top level of the module
def run_api(cfg: DictConfig): # <-- Renamed back to run_api
    """Runs the FastAPI application using Uvicorn."""
    print(f"Starting API with host={cfg.api.host} port={cfg.api.port}")
    uvicorn.run(
        app, # Pass the app instance defined above
        host=cfg.api.host,
        port=cfg.api.port,
        # lifespan="on", # Uvicorn handles lifespan automatically when passed to app
        # reload=True # Only use during development if needed
    )