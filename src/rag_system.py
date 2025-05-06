# src/rag_system.py
import os
from typing import List, Dict
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer # For the generator LLM
import torch # For GPU support

def get_rag_answer(cfg: DictConfig, question: str) -> str:
    """
    Performs RAG to answer a question based on the vector database.
    """
    vector_db_path = cfg.model.vector_db_path
    embedding_model_name = cfg.model.embedding_model_name
    generator_llm_name = cfg.model.generator_llm_name
    collection_name = cfg.project_name.lower().replace(" ", "_") + "_collection"

    # --- 1. Initialize Embedding Model (Needs to be the same as used for indexing) ---
    print("Loading embedding model for querying...")
    try:
        # Ensure device matches indexing if possible, but CPU is safe
        embedding_model = SentenceTransformer(embedding_model_name, device='cpu') # Or 'cuda'
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model {embedding_model_name}: {e}")
        return "Error: Could not load embedding model."

    # --- 2. Connect to Vector Database (ChromaDB) ---
    print(f"Connecting to ChromaDB at {vector_db_path}...")
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        collection = client.get_collection(name=collection_name)
        print(f"Connected to collection: {collection_name}")
    except Exception as e:
        print(f"Error connecting to ChromaDB collection {collection_name}: {e}")
        print("Please ensure the vector index was built successfully.")
        return "Error: Could not connect to vector database."

    # --- 3. Embed the User Question ---
    print("Embedding the user question...")
    try:
        question_embedding = embedding_model.encode(question).tolist()
        print("Question embedded.")
    except Exception as e:
        print(f"Error embedding question: {e}")
        return "Error: Could not embed question."


    # --- 4. Retrieve Relevant Chunks from Vector Database ---
    # We'll retrieve the top N most similar chunks
    num_results = 1 # You can make this configurable
    print(f"Searching vector database for top {num_results} relevant chunks...")
    try:
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=num_results,
                include=['documents', 'distances', 'metadatas','embeddings'] # Include more info for debugging
            )
            print("\n--- ChromaDB Query Results ---") # Add this line
            print(results) # Add this line to see the raw results object
            print("----------------------------") # Add this line

            retrieved_chunks = results['documents'][0] # Extract the list of text chunks
            print(f"Retrieved {len(retrieved_chunks)} relevant chunks.") # This should now print if the query completed

    except Exception as e:
            print(f"Error during vector database search: {e}")
            return "Error: Could not retrieve relevant information."

    if not retrieved_chunks:
        return "Could not find relevant information in the handbook to answer your question."

    # --- 5. Prepare Context for LLM ---
    # Combine the retrieved text chunks into a single context string
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Craft a prompt for the generator LLM
    # This prompt instructs the LLM on how to use the provided context
    prompt = f"""Use the following context from the school handbook to answer the user's question.""""\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
    print("Crafting prompt for generator LLM...")

     # --- 6. Generate the Answer using the Generator LLM ---
    # This section is missing from your code! Add this part.
    print(f"Loading generator LLM: {generator_llm_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(generator_llm_name)
        model = AutoModelForCausalLM.from_pretrained(generator_llm_name)
        print("Generator LLM loaded.")

        # Move model to GPU if available and configured (optional)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(f"Using device for LLM: {device}")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate the response
        # Set reasonable generation parameters
        output_sequences = model.generate(
            inputs["input_ids"],
            max_length=500, # Max length of the generated response
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            attention_mask=inputs["attention_mask"]
        )

        # Decode the generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # Find and extract the answer part from the generated text
        answer_start_marker = "Answer:"
        answer_start_index = generated_text.find(answer_start_marker)

        if answer_start_index != -1:
             final_answer = generated_text[answer_start_index + len(answer_start_marker):].strip()
        else:
             final_answer = generated_text.strip() # Fallback

        print("Answer generated.")
        return final_answer # <--- The function returns the answer here

    except Exception as e:
        print(f"Error generating answer with LLM {generator_llm_name}: {e}")
        return "Error: Could not generate answer."
    
    # Example of how you might test this function (optional, can remove later)
# This code only runs when you execute src/rag_system.py directly
# if __name__ == "__main__":
#     from omegaconf import OmegaConf

#     try: # <--- Add this try block
#         # --- Put the dummy_cfg object definition here ---
#         dummy_cfg = OmegaConf.create({
#             'project_name': 'IntelligentDocumentQASystem',
#             'model': {
#                 'vector_db_path': 'data/vector_db', # Use your actual DB path
#                 'embedding_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
#                 'generator_llm_name': 'distilbert/distilgpt2', # Or larger
#             },
#             'data': {
#                 'processed_data_path': 'data/processed'
#             }
#         })
#         # --- End of dummy_cfg object definition ---

#         # --- Call the function using dummy_cfg ---
#         test_question = "What is the school's policy on bullying?"
#         print(f"Asking question: '{test_question}'")
#         answer = get_rag_answer(dummy_cfg, test_question)
#         print("\n--- Generated Answer ---")
#         print(answer)

#     except Exception as e: # <--- Add this except block
#         print("\n--- An unexpected error occurred during test execution ---")
#         print(f"Error Type: {type(e).__name__}")
#         print(f"Error Details: {e}")
#         import traceback
#         traceback.print_exc() # Print the full traceback