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
import docx
import pypandoc

# Load environment variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Configuration constants
CONFIG = {
    "chunk_size": 3500,
    "chunk_overlap": 2000,
    "embedding_model": "text-embedding-ada-002",
    "retriever_k": 3,
    "extraction_temperature": 0,
    "creative_temperature": 0.3,
    "general_temperature": 0.5,
}

DEFAULT_ANSWER_PROMPT = """Using the following context, provide a detailed and well-structured answer to the question. 
Be creative in your expression while maintaining accuracy with the context.
Rephrase and reorganize the information to make it more engaging.

Question: {question}

Context: {context}

Instructions:
1. Strictly follow the document structure like table etc

Answer:\n"""

DEFAULT_EXTRACTION_PROMPT = """
[INST] Based on the content of the document, find all the Assessment Task 1: Knowledge questions without any modifications. Format your response as a numbered list. [/INST]
"""


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "extracted_questions" not in st.session_state:
        st.session_state.extracted_questions = None
    if "generated_answers" not in st.session_state:
        st.session_state.generated_answers = {}
    if "generate_answer_prompt" not in st.session_state:
        st.session_state.generate_answer_prompt = DEFAULT_ANSWER_PROMPT
    if "answers_generated" not in st.session_state:
        st.session_state.answers_generated = False
    if "kb_qa" not in st.session_state:
        st.session_state.kb_qa = None
    if "guidelines_qa" not in st.session_state:
        st.session_state.guidelines_qa = None


def save_answers_to_markdown(questions, answers, filename="assessment_answers.md"):
    """Save questions and answers to a markdown file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_md:
        temp_md.write("# Assessment Task 1: Knowledge Questions and Answers\n\n")

        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            temp_md.write(f"## Question {question}\n\n")
            temp_md.write(f"**Answer:** {answer}\n\n")

        return temp_md.name


def save_markdown_to_docx(markdown_path, output_filename="assessment_answers.docx"):
    """Convert markdown file to DOCX using Pandoc."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
        try:
            pypandoc.convert_file(markdown_path, "docx", outputfile=temp_docx.name)
            return temp_docx.name
        except Exception as e:
            st.error(f"Error converting markdown to DOCX: {e}")
            return None


def save_answers(questions, answers):
    """Main function to save answers first as markdown and then convert to DOCX."""
    # Convert dictionary to lists maintaining question order
    answer_list = [answers[q] for q in questions]
    markdown_path = save_answers_to_markdown(questions, answer_list)
    return save_markdown_to_docx(markdown_path)


def load_document(uploaded_file):
    """Load and process an uploaded document."""
    if uploaded_file is None:
        return []

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        loader = UnstructuredWordDocumentLoader(tmp_file_path)
        documents = loader.load()
        os.remove(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return []


def setup_qa_chain(documents, collection_name):
    """Set up the question-answering chain with given documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=CONFIG["embedding_model"]),
        collection_name=collection_name,
    )

    llm = ChatOpenAI(temperature=CONFIG["extraction_temperature"])
    retriever = vector_db.as_retriever(search_kwargs={"k": CONFIG["retriever_k"]})

    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )


def find_relevant_context(qa_chain, question):
    """Find relevant context for a question using the QA chain."""
    result = qa_chain.invoke(
        {
            "query": f"""Find relevant information for this question from the knowledge base: {question}
        If you find relevant information, return it.
        If you don't find relevant information, respond with 'NO_CONTEXT_FOUND'."""
        }
    )
    return result["result"]


def generate_answer(qa_chain, question, guidelines_qa=None, custom_prompt=None):
    """Generate an answer for a given question using available context."""
    context = find_relevant_context(qa_chain, question)

    if "NO_CONTEXT_FOUND" in context and guidelines_qa:
        context = find_relevant_context(guidelines_qa, question)

    if "NO_CONTEXT_FOUND" not in context:
        creative_llm = ChatOpenAI(temperature=CONFIG["creative_temperature"])
        response = creative_llm.invoke(
            custom_prompt.format(question=question, context=context)
        )
        return response.content

    general_llm = ChatOpenAI(temperature=CONFIG["general_temperature"])
    response = general_llm.invoke(
        f"""Please answer this question to the best of your ability.
        Be informative yet engaging in your response.
        
        Question: {question}
        
        Note: This answer is based on general knowledge as no specific context was found in the provided documents.
        
        Answer:"""
    )
    return response.content


def extract_questions(qa_chain, prompt):
    """Extract questions from the document using the QA chain."""
    result = qa_chain.invoke({"query": prompt})
    questions_text = result["result"]
    return [q.strip() for q in questions_text.split("\n") if q.strip()]


def render_answer_generation(progress_bar, answers_container):
    """Generate answers progressively and display them immediately."""
    total_questions = len(st.session_state.extracted_questions)

    for idx, question in enumerate(st.session_state.extracted_questions):
        # Update progress bar first
        progress = (idx + 1) / total_questions
        progress_bar.progress(progress)

        # Generate answer
        formatted_prompt = st.session_state.generate_answer_prompt.format(
            question=question, context="{context}"
        )
        answer = generate_answer(
            st.session_state.kb_qa,
            question,
            st.session_state.guidelines_qa,
            formatted_prompt,
        )

        # Store the answer
        st.session_state.generated_answers[question] = answer

        # Display the answer immediately in the main content area
        with answers_container.expander(f"Question: {question}", expanded=False):
            st.write(answer)

    # Mark generation as complete
    st.session_state.answers_generated = True


def render_sidebar():
    """Render all input controls in the sidebar."""
    # Initialize return values
    knowledge_base = None
    guidelines = None

    with st.sidebar:
        st.header("Configuration")

        # Step 1: Question Extraction
        with st.expander("Step 1: Extract Questions", expanded=True):
            questions_doc = st.file_uploader(
                "Upload questions document", type=["docx"], key="questions_doc"
            )

            if questions_doc is not None:
                custom_prompt = st.text_area(
                    "Customize extraction prompt", DEFAULT_EXTRACTION_PROMPT, height=100
                )
                if st.button("Extract Questions"):
                    documents = load_document(questions_doc)
                    qa_chain = setup_qa_chain(documents, "questions-store")
                    st.session_state.extracted_questions = extract_questions(
                        qa_chain, custom_prompt
                    )
                    st.success("Questions extracted successfully!")
                    st.session_state.answers_generated = False

        # Step 2: Reference Documents
        if st.session_state.extracted_questions:
            with st.expander("Step 2: Upload Reference Documents", expanded=True):
                knowledge_base = st.file_uploader(
                    "Upload Knowledge Base", type=["docx"], key="knowledge_base"
                )
                guidelines = st.file_uploader(
                    "Upload Guidelines", type=["docx"], key="guidelines"
                )

            # Prompt Customization
            with st.expander("Promt Modification (optional)", expanded=False):
                custom_prompt = st.text_area(
                    "Customize answer generation prompt:",
                    value=st.session_state.generate_answer_prompt,
                    height=150,
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Prompt"):
                        st.session_state.generate_answer_prompt = custom_prompt
                        st.success("Saved!")
                with col2:
                    if st.button("Reset Default"):
                        st.session_state.generate_answer_prompt = DEFAULT_ANSWER_PROMPT
                        st.success("Reset!")

            # Generate Answers Button in sidebar
            if knowledge_base:
                if st.button("Generate Answers", type="primary", key="generate_btn"):
                    # Setup QA chains
                    st.session_state.kb_qa = setup_qa_chain(
                        load_document(knowledge_base), "kb-store"
                    )
                    if guidelines:
                        st.session_state.guidelines_qa = setup_qa_chain(
                            load_document(guidelines), "guidelines-store"
                        )
                    else:
                        st.session_state.guidelines_qa = None

                    st.session_state.trigger_generation = True

    return knowledge_base, guidelines


def main():
    st.title("IntelliAnswer QnA")
    initialize_session_state()

    # Render sidebar and get uploaded files
    knowledge_base, guidelines = render_sidebar()

    # Main content area
    if st.session_state.extracted_questions:
        st.header("Extracted Questions")
        with st.expander("View All Questions", expanded=True):
            for question in st.session_state.extracted_questions:
                st.write(f"{question}")

        st.markdown("---")

        # Create containers for progress and answers in main area
        if not st.session_state.answers_generated:
            st.header("Generated Answers")
            progress_bar = st.progress(0)
            answers_container = st.container()

            # Check if generation has been triggered from sidebar
            if (
                "trigger_generation" in st.session_state
                and st.session_state.trigger_generation
            ):
                st.session_state.trigger_generation = False  # Reset trigger
                render_answer_generation(progress_bar, answers_container)

        # Show download button after answers are generated
        if st.session_state.answers_generated:
            if st.button("Save Answers as DOCX", type="primary"):
                try:
                    saved_file_path = save_answers(
                        st.session_state.extracted_questions,
                        st.session_state.generated_answers,
                    )

                    with open(saved_file_path, "rb") as file:
                        st.download_button(
                            label="Download Answers",
                            data=file.read(),
                            file_name="assessment_answers.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    st.success("Answers saved successfully!")
                except Exception as e:
                    st.error(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
