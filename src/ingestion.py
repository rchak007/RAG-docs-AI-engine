import os
import pandas as pd
from langchain.document_loaders import UnstructuredWordDocumentLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, CSVLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_documents(data_folder, chunk_size=1000, chunk_overlap=200):
    """
    Parse all documents (Word and Excel) from the given folder.
    """
    logging.info(f"Loading files from {data_folder}")
    docs = []

    # Define text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Parse Word files
    for file in os.listdir(data_folder):
        filepath = os.path.join(data_folder, file)
        logging.info(f"Processing file: {file}")
        if file.endswith(".docx"):          
            logging.info(f"Processing Word file: {filepath}")
            # loader = UnstructuredWordDocumentLoader(os.path.join(data_folder, file))
            # docs.extend(loader.load())
            loader = UnstructuredWordDocumentLoader(filepath, mode="single")
            docs_from_loader = loader.load()

            # Split documents into chunks
            chunks = text_splitter.split_documents(docs_from_loader)
            for chunk in chunks:
                docs.append(Document(page_content=chunk.page_content, metadata={"source": file}))


            # Append the larger chunks to the final document list with metadata
            # for chunk in larger_chunks:
            #     docs.append(Document(page_content=chunk.page_content, metadata={"source": file}))
            
            # for doc in loader.load():
            #     docs.append(Document(page_content=doc.page_content, metadata={"source": file}))


    # Parse Excel files
    # for file in os.listdir(data_folder):
        # logging.info(f"Processing file: {file}")
        elif file.endswith(".xlsx"):
            # df = pd.read_excel(os.path.join(data_folder, file))
            # for _, row in df.iterrows():
            #     docs.append({"content": row.to_dict()})
                        # Combine all rows into a single document
            logging.info(f"Processing Excel file: {filepath}")            
            df = pd.read_excel(filepath)
            combined_content = "\n".join(df.astype(str).apply(" | ".join, axis=1))  # Concatenate rows
            chunks = text_splitter.split_text(combined_content)
            for i, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={"source": file, "chunk": i}))

            # docs.append({"content": combined_content})  # Treat the entire file as one document
            # docs.append(Document(page_content=combined_content, metadata={"source": file}))


    logging.info(f"Total documents parsed: {len(docs)}")
    return docs

if __name__ == "__main__":
    data_folder = "/home/rchak007/github/sapLLM/data"  # Adjust path as needed
    documents = parse_documents(data_folder)
    print(f"Loaded {len(documents)} documents.")
