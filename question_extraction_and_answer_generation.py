import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def load_document(uploaded_file):
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            loader = UnstructuredWordDocumentLoader(tmp_file_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")
            os.remove(tmp_file_path)  # Clean up temp file
            return documents
        except Exception as e:
            st.error(f"Error loading document: {e}")
            return []
    else:
        return []


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=2000)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, collection_name="local-rag"):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        collection_name=collection_name,
    )
    return vector_db


def setup_language_model():
    return ChatOpenAI(temperature=0)


def create_qa_pipeline(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain


def extract_questions(qa_chain, prompt):
    result = qa_chain.invoke({"query": prompt})
    # Convert the result to a list of questions
    questions_text = result["result"]
    # Simple parsing assuming questions are numbered
    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
    return questions


def generate_answer(qa_chain, question, guidelines_qa=None):
    # First try to find answer from knowledge base
    result = qa_chain.invoke(
        {
            "query": f"""Answer this question using the provided knowledge base: {question}
        If you find relevant information, provide a detailed answer.
        If you don't find relevant information, respond with 'NO_CONTEXT_FOUND'."""
        }
    )

    answer = result["result"]

    # If no context found in knowledge base, try guidelines
    if "NO_CONTEXT_FOUND" in answer and guidelines_qa:
        result = guidelines_qa.invoke(
            {
                "query": f"""Answer this question using the provided guidelines: {question}
            If you find relevant information, provide a detailed answer.
            If you don't find relevant information, respond with 'NO_CONTEXT_FOUND'."""
            }
        )
        answer = result["result"]

        # If still no context found, use LLM's general knowledge
        if "NO_CONTEXT_FOUND" in answer:
            result = qa_chain.invoke(
                {
                    "query": f"""Please answer this question to the best of your ability using your general knowledge: {question}
                Note: This answer is based on general knowledge as no specific context was found in the provided documents."""
                }
            )
            answer = result["result"]

    return answer

def main():
    st.title("Assessment Task 1: Knowledge Questions Solver")

    # Initialize session state for storing questions
    if "extracted_questions" not in st.session_state:
        st.session_state.extracted_questions = None

    # Question Extraction Section
    st.header("Step 1: Extract Questions")
    questions_doc = st.file_uploader(
        "Upload questions document", type=["docx"], key="questions_doc"
    )

    if questions_doc is not None:
        documents = load_document(questions_doc)
        splits = split_documents(documents)
        vectorstore = create_vector_store(splits, "questions-store")
        llm = setup_language_model()
        qa_chain = create_qa_pipeline(vectorstore, llm)

        default_prompt = """
        [INST] Based on the content of the document, find Assessment Task 1: Knowledge questions without any modifications. Format your response as a numbered list. [/INST]
        """
        custom_prompt = st.text_area("Customize the prompt", default_prompt, height=100)

        if st.button("Extract Questions"):
            st.session_state.extracted_questions = extract_questions(
                qa_chain, custom_prompt
            )
            st.success("Questions extracted successfully!")

    # Always show extracted questions if they exist
    if st.session_state.extracted_questions:
        with st.expander("Show Extracted Questions", expanded=True):
            for i, question in enumerate(st.session_state.extracted_questions, 1):
                st.write(f"{question}")
        st.markdown("---")

        # Answer Generation Section
        st.header("Step 2: Upload Reference Documents")

        col1, col2 = st.columns(2)
        with col1:
            knowledge_base = st.file_uploader(
                "Upload Knowledge Base", type=["docx"], key="knowledge_base"
            )
        with col2:
            guidelines = st.file_uploader(
                "Upload Guidelines", type=["docx"], key="guidelines"
            )

        if st.button("Generate Answers"):
            if not knowledge_base:
                st.error("Please upload at least the knowledge base document.")
                return

            # Process knowledge base
            kb_documents = load_document(knowledge_base)
            kb_splits = split_documents(kb_documents)
            kb_vectorstore = create_vector_store(kb_splits, "kb-store")
            llm = setup_language_model()
            kb_qa = create_qa_pipeline(kb_vectorstore, llm)

            # Process guidelines if provided
            guidelines_qa = None
            if guidelines:
                guide_documents = load_document(guidelines)
                guide_splits = split_documents(guide_documents)
                guide_vectorstore = create_vector_store(
                    guide_splits, "guidelines-store"
                )
                guidelines_qa = create_qa_pipeline(guide_vectorstore, llm)

            # Generate answers for each question
            st.subheader("Generated Answers:")
            for i, question in enumerate(st.session_state.extracted_questions, 1):
                with st.expander(f"Question: {question}"):
                    answer = generate_answer(kb_qa, question, guidelines_qa)
                    st.write(answer)


if __name__ == "__main__":
    main()
