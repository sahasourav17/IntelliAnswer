# pip install --upgrade nltk
# !pip install langchain unstructured  python-docx sentence-transformers transformers torch accelerate
# !pip install langchain-community
# !pip install "unstructured[docx,pptx]"


import nltk
import time
import os

nltk.download("punkt")
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_openai import ChatOpenAI


def load_docx(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500)
    chunks = text_splitter.split_documents(documents)
    document = chunks[0]
    print(document.page_content)
    print(document.metadata)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING"),
        ),
        collection_name="local-rag",
    )

    return vector_db


def setup_llm():
    # local_model = "llama3.1:latest"
    llm = AzureChatOpenAI(
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    return llm


def create_question_extraction_pipeline(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain


def extract_questions(qa_chain):
    query = """
    [INST] Based on the content of the document, find all the questions for assesment task 1. 
    Format your response as a numbered list. [/INST]
    """
    result = qa_chain({"query": query})
    return result["result"]


def main():
    file_path = "BSBFIN501 Student Assessment Tasks.docx"

    documents = load_docx(file_path)
    splits = split_documents(documents)
    # for i in range(len(splits)):
    #   print(f'======splits[{i}]========== {splits[i]}\n\n')
    vectorstore = create_vector_store(splits)
    llm = setup_llm()  # need to use opeanai here

    start_time = time.time()
    qa_chain = create_question_extraction_pipeline(vectorstore, llm)

    questions = extract_questions(qa_chain)
    end_time = time.time()
    print("Extracted Questions:")
    print(questions)
    print(f"Time spent: {end_time-start_time} seconds")


if __name__ == "__main__":
    main()
