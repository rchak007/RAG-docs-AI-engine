### **High-Level Overview**

Building a document retrieval system for SAP HR Time Management requirements documents using an embedding-based search approach. It processes documents like Word and Excel, embeds them into a vector space for semantic similarity, and allows querying using GPT-4 for advanced reasoning over the retrieved content.

------

### **Detailed Breakdown**

#### **1. Ingestion**

- "Started by **parsing documents** from a folder containing Word and Excel files.

- The system loads each file, chunks the content into manageable pieces (e.g., 1000 characters with 200-character overlap), and adds metadata like file names to track sources."

- Tech Highlights

  :

  - Used `UnstructuredWordDocumentLoader` for loading Word docs.
  - Used Pandas to process Excel sheets by concatenating rows into coherent text.

------

#### **2. Embedding**

- "After parsing, needed to **convert text into numerical vectors** so the system could retrieve semantically similar content. Used the `sentence-transformers/all-mpnet-base-v2` model, which is one of the best for semantic similarity."

- Batch Processing

  :

  - "Processed the documents in batches of 100 to avoid memory issues."
  - The embeddings were stored in a **FAISS vector database**, which enables efficient similarity searches.

- Why Embedding?

  :

  - "Instead of keyword search, embeddings allow me to search based on meaning. For example, if I search for 'Vacation Quota rules,' it can retrieve related chunks even if they use synonyms or related concepts."

------

#### **3. Querying**

- "For querying, combined a 

  retrieval-augmented generation (RAG) approach:

  1. **Retrieve**: I search the FAISS index to fetch the top-k most relevant document chunks.
  2. **Reason**: I pass these chunks to GPT-4 for reasoning. GPT-4 uses the retrieved context to generate accurate answers."

- Example Workflow:

  - "If we ask: 'What are the vacation quota eligibility rules?':
    1. FAISS retrieves chunks about vacation and eligibility.
    2. GPT-4 processes these chunks and crafts a response."

- Flexibility:

  - "I can swap GPT-4 with GPT-3.5 for cost-saving or add my own prompts to refine GPT's reasoning."

------

#### **4. Knowledge Enhancements**

- "I also created a knowledge document to provide additional context. For example:
  - Definitions of key terms like Personnel Area (PA), Employee Group (EG), and eligibility patterns like 'PA=1* means all areas starting with 1.'"
- Why?:
  - "This ensures that the retrieval system can capture domain-specific nuances, especially for SAP HR."

------

### **How It All Comes Together**

1. **Ingestion**: Raw documents are loaded and chunked into pieces with metadata.

2. **Embedding**: Chunks are converted into numerical vectors and stored in FAISS.

3. Querying

   :

   - **Retrieve**: Search for relevant chunks based on a user query.
   - **Reason**: Use GPT-4 to provide natural language answers based on the chunks.

4. **Persistent Knowledge**: All documents are embedded once and reused for querying, so I don’t need to re-ingest unless I add new files.

------

### **Challenges and Lessons Learned**

- **Handling Large Data**: "I optimized chunk sizes and batch sizes to balance performance and memory usage."
- **GPT-4 Costs**: "Each query costs money, so I added safeguards like previewing retrieved content before invoking GPT-4."
- **Better Results**: "Switching from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` significantly improved retrieval accuracy."

------

### upside

"This system bridges static documents and advanced AI reasoning. Instead of just searching for keywords, it understands the meaning behind queries and retrieves relevant rules or policies—even for something as niche as SAP HR."
