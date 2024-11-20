import json
import requests
import fitz  # PyMuPDF for PDF handling
from PIL import Image  # For working with images
import pytesseract  # OCR library
from llama_index.core import VectorStoreIndex, Document  # For vector storage and document indexing
from llama_index.embeddings.ollama import OllamaEmbedding  # Ollama embedding for text representation
from llama_index.llms.ollama import Ollama  # Ollama LLM for classification

# Load configuration from JSON file for easier customization and parameter management
#change it to config.json, while running the code
with open("personalconfig.json", "r") as config_file:
    config = json.load(config_file)

# Configure external tools and APIs using data from the config file
pytesseract.pytesseract.tesseract_cmd = config["tesseract"]["path"]  # Path to Tesseract OCR installation

# Configure embedding model and LLM using Ollama API
ollama_embedding = OllamaEmbedding(
    model_name=config["ollama"]["embedding_model"],  # Embedding model name
    base_url=config["ollama"]["embedding_base_url"]  # Base URL for embedding service
)

llm = Ollama(
    model=config["ollama"]["llm_model"],  # LLM model version
    request_timeout=config["ollama"]["llm_request_timeout"]  # Timeout for API calls
)

def fetch_labels_from_osdu(base_url, headers, kind, limit):
    """
    Fetch labels for document classification from the OSDU platform.

    Args:
        base_url (str): Base URL for the OSDU API.
        headers (dict): API headers including authorization.
        kind (str): Resource kind for fetching labels.
        limit (int): Maximum number of records to fetch per query.

    Returns:
        tuple: Two dictionaries - one mapping 'Code' to 'id' and another for descriptions.
    """
    query_url = f"{base_url}/api/search/v2/query_with_cursor"  # API endpoint for querying data
    query_payload = {
        "kind": kind,  # Specify the resource kind to fetch
        "limit": limit,  # Pagination limit
        "returnedFields": ["data.Code", "data.Description", "id"]  # Fields to retrieve
    }

    labels = {}  # Map to store 'Code' to 'id'
    labels_desc = {}  # Map to store descriptions
    has_more = True  # Flag for pagination control

    while has_more:
        response = requests.post(query_url, headers=headers, json=query_payload)  # API request
        response.raise_for_status()  # Raise error if request fails
        data = response.json()  # Parse response as JSON

        # Process results to populate label dictionaries
        for record in data.get("results", []):
            code = record.get("data", {}).get("Code")
            record_id = record.get("id")
            description = record.get("data", {}).get("Description")
            if code and record_id:
                labels[code] = record_id
                labels_desc[code] = description

        # Check if more results are available using cursor
        next_cursor = data.get("cursor")
        if next_cursor:
            query_payload["cursor"] = next_cursor  # Update payload for next query
        else:
            has_more = False  # Exit loop if no more results

    return labels, labels_desc

def extract_text_from_pdf(pdf_path, max_pages=None):
    """
    Extract text content from a PDF file using PyMuPDF and OCR for non-textual pages.

    Args:
        pdf_path (str): Path to the PDF file.
        max_pages (int, optional): Maximum number of pages to process.

    Returns:
        str: Combined text from all processed pages.
    """
    text = ""  # Initialize empty text container
    pdf_document = fitz.open(pdf_path)  # Open PDF file
    num_pages = pdf_document.page_count  # Total pages in PDF
    pages_to_extract = min(num_pages, max_pages) if max_pages else num_pages  # Limit pages if specified

    for page_num in range(pages_to_extract):
        page = pdf_document[page_num]  # Access a specific page
        page_text = page.get_text()  # Extract text from the page
        if page_text.strip():
            text += page_text  # Append extracted text
        else:
            # Use OCR if no text is found
            pix = page.get_pixmap()  # Render page to image
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image
            page_text = pytesseract.image_to_string(pil_image)  # Perform OCR
            text += page_text + "\n"  # Append OCR result

    pdf_document.close()  # Close the PDF file
    return text  # Return combined text

def classify_text_with_llm(text, label_dict, index):
    """
    Classify text using the LlamaIndex retriever and Ollama LLM.

    Args:
        text (str): Input text for classification.
        label_dict (dict): Dictionary of labels for classification.
        index (VectorStoreIndex): Document index for label retrieval.

    Returns:
        None: Prints the classification result.
    """
    retriever = index.as_retriever()  # Create retriever from index
    query_response = retriever.retrieve(text)  # Retrieve closest labels based on text

    # Build classification prompt for LLM
    prompt = f"Classify the following text from the document: '{text}'\nPossible categories:\n"
    for i, result in enumerate(query_response):
        prompt += f"{i + 1}. {result.metadata['key']}: {result.text} (OSDU record id: {result.metadata['id']})\n"

    prompt += "Which category does it best belong to?"  # Finalize prompt

    # Stream and print LLM response
    response = llm.stream_complete(prompt)
    for r in response:
        print(r.delta, end="")

def main():
    """
    Main workflow for extracting, classifying, and mapping text from a PDF
    against labels fetched from OSDU.

    Steps:
        1. Fetch classification labels from OSDU.
        2. Create document index using labels.
        3. Extract text from PDF using OCR or direct extraction.
        4. Classify text using an LLM-based retriever.
    """
    osdu_config = config["osdu"]  # Load OSDU API configuration
    pdf_config = config["pdf"]  # Load PDF-related settings
    query_config = config["osdu_query"]  # Load OSDU query settings

    # Fetch labels from OSDU
    label_dict, label_desc = fetch_labels_from_osdu(
        osdu_config["base_url"], osdu_config["headers"],
        query_config["kind"], query_config["limit"]
    )

    # Create documents for classification
    documents = [
        Document(text=label_desc[label], metadata={'key': label, 'id': label_dict[label]})
        for label in label_dict
    ]
    index = VectorStoreIndex.from_documents(documents, embed_model=ollama_embedding)  # Build index

    # Extract text from the specified PDF file
    text = extract_text_from_pdf(pdf_config["sample_path"], pdf_config["max_pages"])

    # Classify the extracted text using the document index and LLM
    classify_text_with_llm(text, label_dict, index)

if __name__ == "__main__":
    main()  # Run the main workflow
