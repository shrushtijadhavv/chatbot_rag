from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, UnstructuredWordDocumentLoader

import os

# 🔁 Load all file types
def load_documents(folder_path="documents/"):
    docs = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(filepath)

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)

        elif filename.endswith(".csv"):
            loader = CSVLoader(filepath)

        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)

        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        docs.extend(loader.load())

    return docs

# Split, embed, and store in ChromaDB
docs = load_documents()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

vectordb.persist()
print("All documents embedded and saved to ChromaDB.")
