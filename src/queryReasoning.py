from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

def query_with_reasoning(index_path, query, top_k=5):
    """
    Retrieve results from FAISS and reason over them using an LLM.
    """
    # Load FAISS index
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Retrieve similar documents
    results = vectorstore.similarity_search(query, k=top_k)
    if not results:
        print("No relevant context found for the query. Please refine your question.")
        return

    # Combine retrieved chunks
    combined_context = "\n\n".join([doc.page_content for doc in results])
    print("Retrieved Context:")
    # print(combined_context)

    # Query the LLM with context
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    You are an expert in SAP HR Time Management.
    Below is the retrieved context from documents:
    {combined_context}

    Based on the context above, answer the following question as accurately and completely as possible:
    {query}

    If the context does not contain enough information, clearly state that the information is not available.
    """


    response = llm.invoke(prompt)
    print(f"Query: {query}\n")
    print(f"Response:\n{response.content}")



# def query_vectorstore(index_path, query, top_k=5):
#     """
#     Query the FAISS vector store for the most similar documents to the query.
#     """
#     # Load the FAISS index
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     # vectorstore = FAISS.load_local(index_path, embeddings)
#     vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


#     # Perform the similarity search
#     results = vectorstore.similarity_search(query, k=top_k)
#     if not results:
#         print("No relevant context found for the query. Please refine your question.")
#         return

#     # Print the top results
#     print(f"Query: {query}\n")
#     for i, result in enumerate(results, 1):
#         print(f"Result {i}:")
#         print(result.page_content)
#         print("---")


# if __name__ == "__main__":
#     # Path to the FAISS index
#     index_path = "faiss_index"

#     # Query example
#     # query = "What are the absence quota rules for employees in Personnel Area A?"
#     query = "What are the absence quota eligiblity rules for Vacation Quota"
    
#     # Run the query
#     query_vectorstore(index_path, query, top_k=5)

if __name__ == "__main__":
    index_path = "faiss_index"
    query = "What are the absence quota eligibility rules for Vacation Quota?"
    query_with_reasoning(index_path, query, top_k=10)    
