import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def criaRetriever():
    loader = WebBaseLoader(
        web_paths=[
            "https://focanasmidias.com.br/post/",
            "https://focanasmidias.com.br/sobre",
            "https://focanasmidias.com.br/"
        ]
    )

    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documentos)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embedding=embeddings
    )

    return vectorstore.as_retriever()
