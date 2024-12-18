from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os


def add_new_document(index_path, new_document_content):
    """
    Add a new document to an existing FAISS index.
    """
    # Load the existing FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings)
        print("Loaded existing FAISS index.")
    else:
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run embeddings.py first.")

    # Embed the new document
    new_doc = Document(page_content=new_document_content)
    new_vectorstore = FAISS.from_documents([new_doc], embedding=embeddings)
    print("New document embedded.")

    # Merge the new embedding into the existing index
    vectorstore.merge_from(new_vectorstore)

    # Save the updated FAISS index
    vectorstore.save_local(index_path)
    print("New document added and FAISS index updated.")


if __name__ == "__main__":
    # Path to your FAISS index
    index_path = "faiss_index"

    # Replace this with the content of the new document
    new_document_content = "This is the content of the new document you want to add."

    # Add the new document to the FAISS index
    add_new_document(index_path, new_document_content)
