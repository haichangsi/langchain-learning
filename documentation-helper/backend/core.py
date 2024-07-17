from typing import Any, Dict, List
from dotenv import load_dotenv
import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"), embedding=embddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_document_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_document_chain
    )
    res = qa.invoke(input={"input": query, "chat_history": chat_history})

    # res_mapping = {
    #     "query": res["input"],
    #     "result": res["answer"],
    #     "source_documents": res["context"],
    # }

    return res


if __name__ == "__main__":
    query = "What is a Langchain chain?"
    result = run_llm(query)
    # check source using [doc.metadata["source"] for doc in result["context"]]
    print(result["result"])
