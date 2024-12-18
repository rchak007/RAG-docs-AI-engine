from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def query_vectorstore(index_path, query, top_k=5):
    """
    Query the FAISS vector store for the most similar documents to the query.
    """
    # Load the FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vectorstore = FAISS.load_local(index_path, embeddings)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


    # Perform the similarity search
    results = vectorstore.similarity_search(query, k=top_k)

    # Print the top results
    print(f"Query: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(result.page_content)
        print("---")


if __name__ == "__main__":
    # Path to the FAISS index
    index_path = "faiss_index"

    # Query example
    # query = "What are the absence quota rules for employees in Personnel Area A?"
    query = "What are the absence quota eligiblity rules for Vacation Quota"
    
    # Run the query
    query_vectorstore(index_path, query, top_k=5)
