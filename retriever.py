from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

def retrieve_context(query):
    results = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in results])