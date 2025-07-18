import requests
import json
import os

# Configuration
BASE_URL = "http://localhost:9721"
UPLOAD_ENDPOINT = f"{BASE_URL}/documents/upload"
QUERY_ENDPOINT = f"{BASE_URL}/query"

def upload_document(filepath):
    """Uploads a document to the MiniRAG server."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    with open(filepath, 'rb') as f:
        files = {'file': (os.path.basename(filepath), f)}
        try:
            response = requests.post(UPLOAD_ENDPOINT, files=files)
            response.raise_for_status()  # Raise an exception for bad status codes
            print("Document uploaded successfully.")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error uploading document: {e}")
            return None

def query_server(query_text):
    """Sends a query to the MiniRAG server."""
    payload = {
        "query": query_text,
        "mode": "light",
        "stream": False,
        "only_need_context": False
    }
    try:
        response = requests.post(QUERY_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying server: {e}")
        return None

if __name__ == "__main__":
    # Create a dummy file to upload
    dummy_filepath = "sample_document.txt"
    with open(dummy_filepath, "w") as f:
        f.write("This is a sample document about the history of artificial intelligence.")

    # 1. Upload the document
    upload_result = upload_document(dummy_filepath)

    if upload_result:
        # 2. Query the document
        query = "What is this document about?"
        query_result = query_server(query)

        if query_result:
            print("\n--- Query Result ---")
            print(f"Query: {query}")
            print(f"Response: {query_result.get('response')}")
            print("--------------------")

    # Clean up the dummy file
    os.remove(dummy_filepath)
