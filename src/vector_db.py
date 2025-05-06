# src/vector_db.py
import os
import json
from typing import List, Dict
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
import chromadb

def build_vector_index(cfg: DictConfig):
    """
    Generates embeddings for processed text chunks and stores them in ChromaDB.
    """
    processed_data_path = cfg.data.processed_data_path
    vector_db_path = cfg.model.vector_db_path
    embedding_model_name = cfg.model.embedding_model_name
    collection_name = cfg.project_name.lower().replace(" ", "_") + "_collection" # Dynamic collection name

    # --- 1. Load Processed Data ---
    processed_chunks_file = os.path.join(processed_data_path, "processed_handbook_chunks.json")
    if not os.path.exists(processed_chunks_file):
        print(f"Error: Processed data file not found at {processed_chunks_file}")
        print("Please run the data processing pipeline first.")
        return

    print(f"Loading processed chunks from {processed_chunks_file}...")
    with open(processed_chunks_file, 'r', encoding='utf-8') as f:
        processed_chunks = json.load(f)

    if not processed_chunks:
        print("No chunks found in the processed data. Exiting index building.")
        return

    print(f"Loaded {len(processed_chunks)} chunks.")

    # --- 2. Load Embedding Model ---
    print(f"Loading embedding model: {embedding_model_name}...")
    try:
        # Setting device to 'cpu' for broader compatibility,
        # you can change to 'cuda' if you have a compatible GPU
        model = SentenceTransformer(embedding_model_name, device='cpu')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model {embedding_model_name}: {e}")
        print("Please check model name and internet connection.")
        return


    # --- 3. Initialize Vector Database (ChromaDB) ---
    # ChromaDB will store data in the specified path
    print(f"Initializing ChromaDB at {vector_db_path}...")
    client = chromadb.PersistentClient(path=vector_db_path)

    # Get or create a collection
    # Clean existing collection if it exists to rebuild the index
    try:
         client.delete_collection(name=collection_name)
         print(f"Deleted existing collection: {collection_name}")
    except:
         # Collection doesn't exist, no need to delete
         pass
         
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:batch_size": 5000} # Add this metadata parameter
    )
    print(f"Created/Accessed ChromaDB collection: {collection_name}")

    # --- 4. Generate and Store Embeddings ---
    print("Generating embeddings and adding to vector database...")
    chunk_texts = [chunk["text"] for chunk in processed_chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in processed_chunks]
    # Store doc_id as metadata
    metadatas = [{"doc_id": chunk["doc_id"]} for chunk in processed_chunks]

    # Generate embeddings in batches for efficiency (optional but good practice)
    batch_size = 100 # Adjust batch size based on memory
    for i in range(0, len(chunk_texts), batch_size):
        batch_texts = chunk_texts[i:i+batch_size]
        batch_ids = chunk_ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        # Generate embeddings
        batch_embeddings = model.encode(batch_texts).tolist() # .tolist() for ChromaDB

        # Add to ChromaDB
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_texts, # Store the text itself in the DB
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"Added batch {i//batch_size + 1}/{(len(chunk_texts)-1)//batch_size + 1}")

    print("\nVector index built successfully!")
    print(f"Total documents in collection: {collection.count()}")