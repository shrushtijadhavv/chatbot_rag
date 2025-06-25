import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get Hugging Face API token from environment variable
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MISTRAL_ENDPOINT = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500):
    # Simple chunking by sentences
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def retrieve_relevant_chunks(query, chunks, chunk_embeddings, top_k=2):
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_mistral_response(prompt):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False
        }
    }
    response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"].strip()
        else:
            return str(result)
    else:
        return f"Error: {response.status_code} - {response.text}"

def chatbot_rag_response(user_query, pdf_file):
    if pdf_file is not None:
        pdf_text = extract_pdf_text(pdf_file)
        chunks = chunk_text(pdf_text)
        chunk_embeddings = embedder.encode(chunks)
        relevant_chunks = retrieve_relevant_chunks(user_query, chunks, chunk_embeddings, top_k=2)
        context = "\n".join(relevant_chunks)
    else:
        context = "No PDF uploaded. Please upload a PDF to chat about its content."
    prompt = (
        "Based only on the following context, answer the question as accurately and concisely as possible. "
        "If the answer is not present in the context, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{user_query}\n\n"
        "Answer:"
    )
    answer = generate_mistral_response(prompt)
    return answer

iface = gr.Interface(
    fn=chatbot_rag_response,
    inputs=[
        gr.Textbox(label="Ask a question about your PDF"),
        gr.File(label="Upload a PDF", file_types=[".pdf"])
    ],
    outputs="text",
    title="PDF Semantic QA Chatbot",
    description="Upload a PDF and ask questions about its content. The chatbot uses semantic search and a language model to answer."
)

iface.launch()