# documentzeroshotclassification
We will explore how LLM based zero-shot classification works in the Exploration and Production domain with OSDU and demonstrate its application using a Python-based solution.

```markdown
# PDF Text Extraction and Classification Pipeline

This repository provides a Python-based pipeline to extract text from PDFs, classify the content using zero-shot classification, and map it against labels fetched from the OSDU platform. The workflow integrates tools like OCR (Tesseract), document indexing, and a large language model (LLM) for classification.

---

## Features

- **PDF Text Extraction**: Combines direct text extraction using PyMuPDF and OCR for non-textual pages.
- **OSDU Label Fetching**: Retrieves classification labels from the OSDU platform based on a specified `kind`.
- **Text Classification**: Uses LlamaIndex for label indexing and Ollama LLM for text classification.
- **Configurable Setup**: Highly customizable through a `config.json` file.

---

## Prerequisites

### Software Requirements
- Python 3.9 or above
- Install the required Python libraries using:
  ```bash
  pip install -r requirements.txt
  ```

### External Tools
- **Tesseract OCR**: [Installation Instructions](https://github.com/tesseract-ocr/tesseract)
  - Ensure `tesseract` is installed and the path is configured in `config.json`.
- **OSDU API**: Requires access to OSDU endpoints with valid credentials.
- **Ollama API**: Ensure you have access to Ollama embeddings and LLM APIs [Installation Instructions](https://ollama.com/download).

---

## Configuration

All configurable parameters are stored in a `config.json` file for easy management. Update the file with your specific setup before running the script.

### Example `config.json`

```json
{
  "osdu": {
    "base_url": "",
    "headers": {
      "accept": "application/json",
      "data-partition-id": "",
      "Authorization": "Bearer <token>",
      "Content-Type": "application/json",
      "Accept": "*/*"
    }
  },
  "tesseract": {
    "path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  },
  "ollama": {
    "embedding_model": "nomic-embed-text",
    "embedding_base_url": "http://localhost:11434",
    "llm_model": "llama3.2:latest",
    "llm_request_timeout": 120.0
  },
  "pdf": {
    "sample_path": "F:/PyCharmProjects/zeroshotclassification/sampledata/cr_100806_1.pdf",
    "max_pages": 1
  },
  "osdu_query": {
    "kind": "osdu:wks:reference-data--DocumentType:1.0.1",
    "limit": 5
  }
}
```

### Key Configuration Fields

1. **OSDU**:
   - `base_url`: Base URL for the OSDU API (set to your OSDU instance).
   - `headers`: Includes authorization (`Bearer <token>`), partition ID, and content type for API requests.

2. **Tesseract**:
   - `path`: Path to the Tesseract OCR executable (default is `C:\\Program Files\\Tesseract-OCR\\tesseract.exe` on Windows).

3. **Ollama**:
   - `embedding_model`: Model used for text embedding (`nomic-embed-text`).
   - `embedding_base_url`: Local API base URL for embeddings (`http://localhost:11434`).
   - `llm_model`: LLM used for classification (`llama3.2:latest`).
   - `llm_request_timeout`: Timeout (in seconds) for API requests (default is `120` seconds).

4. **PDF**:
   - `sample_path`: Path to the sample PDF file for text extraction (`F:/PyCharmProjects/zeroshotclassification/sampledata/cr_100806_1.pdf`).
   - `max_pages`: Number of pages to process (default is `1`).

5. **OSDU Query**:
   - `kind`: Specifies the resource type for fetching labels (`osdu:wks:reference-data--DocumentType:1.0.1`).
   - `limit`: Maximum number of records fetched per query (`5`).

---

## Running the Pipeline

### Steps to Execute

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `config.json` is updated with your setup.

4. Run the script:
   ```bash
   python main.py
   ```

---

## Output

- **Extracted Text**: The combined text from the PDF is processed and classified.
- **Classification Result**: Prints the classification result based on OSDU labels and Ollama LLM.

---

## Troubleshooting

### Common Issues

1. **Tesseract Not Found**: Ensure `tesseract` is installed and its path is correctly set in `config.json`.
2. **OSDU API Errors**: Verify the `base_url`, `kind`, and authorization token in `config.json`.
3. **Ollama API Issues**: Confirm the embedding and LLM APIs are accessible.

### Debugging Tips

- Use verbose logging in the script to identify issues.
- Verify API connectivity using tools like `curl` or Postman.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for enhancements or bug fixes.

---

## License

This project is licensed under the [MIT License](LICENSE).
```
