# IntelliAnswer

## Goal:

A system that processes user-provided question files and supplementary documents. It extracts questions, answers them using information from the supplementary files when available, and falls back to an LLM for answers when necessary.

## Installation

1. Create a virtual environment (optional but recommended)

   ```bash
   python -m venv llmrag
   ```

2. Install all the dependencies
   ```bash
   pip install -r requirements.txt
   ```

If you want to use Ollama with local models then follow this steps:

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

If you want to run the webapp, you can use the following command. Make sure you have the `OPENAI_API_KEY` set to your `.env` file

```bash
streamlit run QEAG_webapp.py
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
