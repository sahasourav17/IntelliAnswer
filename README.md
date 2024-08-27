# IntelliAnswer

## Goal:

A system that processes user-provided question files and supplementary documents. It extracts questions, answers them using information from the supplementary files when available, and falls back to an LLM for answers when necessary.

### Implemented Features:

- Read docx files.

  If you want to use the pdf file then instead of using the `read_docx` function, you can use the bellow function

  ```python
  from langchain_community.document_loaders import UnstructuredPDFLoader
  def load_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents
  ```

- Extract information based on user query (currently assessment's task 1 questions )

  ```python
  def extract_questions(qa_chain):
    # change the query according to your task
    query = """
    [INST] Based on the content of the document, find all the questions for assesment task 1.
    Format your response as a numbered list. [/INST]
    """
    result = qa_chain({"query": query})
    return result["result"]
  ```

## Installation

1. Create a virtual environment (optional but recommended)

   ```bash
   python -m venv llmrag
   ```

2. Install all the dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Download `Ollama` from here [https://ollama.com/download]

4. Run `ollama` after installing

5. In terminal you need to pull the `llama` and `nomic-embed-text`. Although you can use any of the model available in the ollama repository.

   ```bash
   ollama run llama3
   ollama pull nomic-embed-text
   ```

6. Verify your installation

   ```bash
   ollama list
   ```

7. Now run the python file. For instance, you can use the following command to run the `langchain_ollama_llama3_rag_for_docx.py` script.

   ```bash
   python3 langchain_ollama_llama3_rag_for_docx.py
   ```

**Note:**

- Before running the script, you must specify the filepath in the `main` function.
- If your docx file is large enough, then try to tweak the `chunk_size` and `chunk_overlap` parameters accordingly.

  ```python
  def split_documents(documents):
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
      chunks = text_splitter.split_documents(documents)
      document = chunks[0]
      print(document.page_content)
      print(document.metadata)
      print(f"Split into {len(chunks)} chunks")
      return chunks
  ```
