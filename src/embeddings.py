# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path  # Add this import to manage paths

from ingestion import parse_documents
import os



def create_vectorstore(documents, batch_size=100):
    """
    Create a FAISS vector store for the parsed documents in batches.
    """

    # modify your embeddings program to use the sentence-transformers/all-mpnet-base-v2 model. 
    #               This will improve the retrieval accuracy significantly compared to all-MiniLM-L6-v2 
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = None

    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedding=embeddings)
        else:
            batch_vectorstore = FAISS.from_documents(batch, embedding=embeddings)
            vectorstore.merge_from(batch_vectorstore)


    # Ensure the save path exists
    save_path = Path("faiss_index")
    save_path.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    # FAISS.save_local("faiss_index", vectorstore)
    # FAISS.save_local(str(save_path), vectorstore)  # Pass the folder path as a string
    vectorstore.save_local(str(save_path))

    print("Vector database created with FAISS")


# def create_vectorstore(documents):
#     """
#     Create a FAISS vector store for the parsed documents.
#     """
#     # Load HuggingFace embedding model
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Create a FAISS index from documents
#     vectorstore = FAISS.from_documents(documents, embedding=embeddings)

#     # Save FAISS index to disk
#     FAISS.save_local("faiss_index", vectorstore)
#     print("Vector database created with FAISS")

# def create_vectorstore(documents):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_documents(documents, embedding_function=embeddings)
#     FAISS.save_local("faiss_index", vectorstore)
#     print("Vector database created with FAISS")
    
# def create_vectorstore(documents, persist_directory="./db"):
#     """
#     Create a vector store for the parsed documents.
#     """
#     # Load HuggingFace embedding model
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Store embeddings in ChromaDB
#     # vectorstore = Chroma.from_documents(documents, embedding_function=embeddings, persist_directory=persist_directory)
    
#         # Store embeddings in ChromaDB
#     vectorstore = Chroma.from_documents(
#         documents=documents,
#         embedding=embeddings,  # Use the updated keyword
#         persist_directory=persist_directory
#     )
#     vectorstore.persist()
#     print(f"Vector database created and saved at {persist_directory}")
#     return vectorstore

if __name__ == "__main__":
    # Parse documents from the data folder
    data_folder = "./data"
    documents = parse_documents(data_folder)

    # # Create a vector database
    # db_directory = "./db"  # Change this directory if needed
    # if not os.path.exists(db_directory):
    #     os.makedirs(db_directory)

    # create_vectorstore(documents, persist_directory=db_directory)
    # Create FAISS vector database
    create_vectorstore(documents,  batch_size=100)    
