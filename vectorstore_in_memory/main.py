import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pathlib import Path

load_dotenv()

if __name__ == "__main__":
    paper = "reAct_paper.pdf"
    pdf_path = Path(__file__).parent / paper
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index_reAct_paper")
    new_vector_store = FAISS.load_local(
        "faiss_index_reAct_paper", embeddings, allow_dangerous_deserialization=True
    )
