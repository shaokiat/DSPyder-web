import dspy
from src.retriever import get_hdb_index, HDBRetriever
from src.model import HDBRAG

def test_retriever():
    print("Initializing index...")
    index = get_hdb_index()
    retriever = HDBRetriever(index=index, k=3)
    
    queries = [
        "What is the Enhanced CPF Housing Grant?",
        "MOP requirements for BTO",
        "fresh start housing scheme eligibility"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever(query)
        for i, res in enumerate(results):
            print(f"Result {i+1}: {res.long_text[:200]}...")

if __name__ == "__main__":
    test_retriever()
