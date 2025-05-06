# src/data_processing.py
import os
# import glob # No longer needed if processing a single file
from typing import List, Dict
from omegaconf import DictConfig
import json
import fitz # PyMuPDF

# Remove the read_text_file and process_documents functions that handle directories
# And add a new function for PDF

def read_pdf_text(filepath: str) -> str:
    """Extracts text content from a PDF file page by page."""
    text = ""
    try:
        doc = fitz.open(filepath)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
            text += "\n--- Page Break ---\n" # Optional: Mark page breaks
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        # Handle errors appropriately
    return text

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text into chunks with overlap."""
    # Use the same chunking logic as before, or a more advanced one.
    # If using Langchain:
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     length_function=len,
    # )
    # return text_splitter.split_text(text)

    # Or the basic word-split logic from before:
    words = text.split()
    chunks = []
    # Ensure chunk_overlap doesn't exceed chunk_size
    actual_overlap = min(chunk_overlap, chunk_size - 1) if chunk_size > 0 else 0

    for i in range(0, len(words), chunk_size - actual_overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


def process_handbook_pdf(cfg: DictConfig) -> List[Dict]:
    """Reads a PDF handbook, extracts text, chunks it, and returns processed data."""
    pdf_path = cfg.data.raw_data_path
    processed_path = cfg.data.processed_data_path
    chunk_size = cfg.data.chunk_size
    chunk_overlap = cfg.data.chunk_overlap

    all_processed_chunks = []

    # Ensure processed directory exists
    os.makedirs(processed_path, exist_ok=True)

    print(f"Processing PDF handbook from {pdf_path}...")
    full_text = read_pdf_text(pdf_path)

    if not full_text:
        print("Could not extract text from PDF. Exiting processing.")
        return []

    chunks = chunk_text(full_text, chunk_size, chunk_overlap)

    doc_id = os.path.basename(pdf_path) # Use filename as doc_id

    for i, chunk in enumerate(chunks): # Renamed 'chunk_text' to 'chunk'
        all_processed_chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{i:04d}",
            "text": chunk, # Use the new variable name here
            # ...
        })

    # Save processed data (optional)
    output_file = os.path.join(processed_path, "processed_handbook_chunks.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed_chunks, f, indent=2)

    print(f"Processed {len(all_processed_chunks)} chunks and saved to {output_file}")
    return all_processed_chunks

# Update the function called by run.py
# def process_documents(cfg: DictConfig): # Remove or modify this
#     # Check cfg.data.dataset_name or just call process_handbook_pdf directly
#     return process_handbook_pdf(cfg) # Call the PDF processing function