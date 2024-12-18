import faiss
import numpy as np  # Add this import
# Check FAISS version
print(f"FAISS version: {faiss.__version__}")

# Test FAISS by creating a simple index and searching it
def test_faiss():
    # Create a FAISS index with 2 dimensions
    dimension = 2
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric

    # Add some sample data to the index
    vectors = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    index.add(np.array(vectors, dtype=np.float32))

    print(f"Number of vectors in index: {index.ntotal}")

    # Query the index with a new vector
    query_vector = np.array([[1.0, 2.1]], dtype=np.float32)
    distances, indices = index.search(query_vector, k=2)

    print(f"Query result indices: {indices}")
    print(f"Query result distances: {distances}")

if __name__ == "__main__":
    test_faiss()

