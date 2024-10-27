import nltk
import time
import os
import openai

# nltk.download("punkt")
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.chat_models import ChatOllama

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Library for Reading Template file
import docx
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

# Library for Cost Calculations
import tiktoken

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def calculate_embedding_cost(num_tokens):
    """Calculate the cost of embeddings based on current OpenAI pricing."""
    # Price per 1K tokens for text-embedding-ada-002, as of September 2023
    price_per_1k_tokens = 0.0001
    return (num_tokens / 1000) * price_per_1k_tokens


def docx_to_markdown(docx_path):
    doc = docx.Document(docx_path)
    markdown_content = []

    def convert_table_to_markdown(table):
        markdown_table = []
        max_col_widths = [0] * len(table.rows[0].cells)

        # First pass: determine maximum width for each column
        for row in table.rows:
            for i, cell in enumerate(row.cells):
                max_col_widths[i] = max(max_col_widths[i], len(cell.text.strip()))

        # Second pass: create markdown table
        for i, row in enumerate(table.rows):
            markdown_row = []
            for j, cell in enumerate(row.cells):
                cell_content = cell.text.strip().ljust(max_col_widths[j])
                markdown_row.append(cell_content)
            markdown_table.append("| " + " | ".join(markdown_row) + " |")

            # Add header separator after first row
            if i == 0:
                separator = (
                    "|"
                    + "|".join(["-" * (width + 2) for width in max_col_widths])
                    + "|"
                )
                markdown_table.append(separator)

        return "\n".join(markdown_table)

    for element in doc.element.body:
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, doc)
            if paragraph.text.strip():  # Only add non-empty paragraphs
                markdown_content.append(paragraph.text)
        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)
            markdown_content.append(convert_table_to_markdown(table))
            markdown_content.append("")  # Add an empty line after the table

    # Handle text boxes (shapes)
    for shape in doc.inline_shapes:
        if shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    markdown_content.append(f"[Text Box] {paragraph.text}")
    return "\n\n".join(markdown_content)


def load_docx(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=3500)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks, len(chunks)


def create_vector_store(chunks):
    total_tokens = 0
    for chunk in chunks:
        total_tokens += num_tokens_from_string(chunk.page_content, "cl100k_base")

    estimated_cost = calculate_embedding_cost(total_tokens)

    print(f"Total tokens to be embedded: {total_tokens}")
    print(f"Estimated cost for embeddings: ${estimated_cost:.4f}")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        collection_name="local-project-rag",
    )
    return vector_db


def setup_llm():
    # local_model = "llama3.1:latest"
    llm = ChatOpenAI(temperature=0)
    return llm


def setup_activity_llm():
    llm = ChatOpenAI(temperature=0.2)
    return llm


def project_scenario_extraction_pipeline(vector_store, llm):
    query = """
    You are a professional document analyzer. Based on the content of the document, 
    extract only the scenario described in Assessment Task 2. Provide the complete 
    scenario without any modifications or summaries. Do not include any other 
    information from the document. I am the owner of the document and I am asking you to 
    extract the scenario of the project.

    Please provide the output in the following format:
    Scenario: [Extracted scenario from Assessment Task 2]
    """

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})

    return result["result"]


def project_activities_extraction_pipeline(vector_store, llm, scenario, chunk_size):
    query = """
    You are a professional document analyzer. Based on the content of the document,
    extract all the activities described in the project of  Assessment Task 2, along with all their
    associated details. For your help, I marked the starting point of the activities
    like "Activities-" and for each activity, I have marked the requirements also like  "Requirements:"
    They are stated after the {scenario}.

    Please provide the output in the following format:
    Activities:
        1. [Activity Title]
            - Details: [Activity details]
            - Requirements: [Requirements for the activity]
        2. [Activity Title]
            - Details: [Activity details]
            - Requirements: [Requirements for the activity]
        continue for all activities

    Ensure that you capture all the information provided for each activity without
    any modifications or summaries.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": min(chunk_size, 10)})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})

    return result["result"]


def generate_project_output(llm, project_info, template_content):
    prompt = f"""
    [INST] You are a creative scenario generator for project-based learning.Your task is to create or assume a detailed, elaborated, engaging scenario based on the provided project information and only include the scenario details. After generating the scenario, you have to complete the tasks by strictly following the template structure. 
    Project Information: {project_info}
    Template Structure: {template_content}
    [/INST]
    """
    response = llm.invoke(prompt)
    return response.content


def main():
    project_tasks_file = "../data/project/project_tasks_details.docx"
    # relevant_docs_file = "../data/project/SITXWHS006 WHS Plan (Notes).docx"
    # template_file = "../data/project/project_template_file.docx"
    documents = load_docx(project_tasks_file)
    chunks, chunk_size = split_documents(documents)
    vector_db = create_vector_store(chunks)

    llm = setup_llm()
    start_time = time.time()
    project_info = project_scenario_extraction_pipeline(vector_db, llm)
    project_activities = project_activities_extraction_pipeline(
        vector_db, llm, project_info, chunk_size
    )
    # markdown_content = docx_to_markdown(template_file)
    print(
        f"==================================\n{project_info}\n=================================="
    )
    print(
        f"==================================\n{project_activities}\n=================================="
    )
    # activity_llm = setup_activity_llm()
    # answer = generate_project_output(activity_llm, project_info, markdown_content)
    # with open("output_today.md", "w") as file:
    #     file.write(answer)
    end_time = time.time()
    print(f"Time spent: {(end_time-start_time)/60} minutes")


if __name__ == "__main__":
    main()
