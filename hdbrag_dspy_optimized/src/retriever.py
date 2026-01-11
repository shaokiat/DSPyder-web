import json
from pathlib import Path
import dspy
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage

class HDBRetriever(dspy.Retrieve):
    """A custom DSPy retriever that uses LlamaIndex for vector search."""
    def __init__(self, index, k=3):
        super().__init__(k=k)
        self.index = index
        self.query_engine = index.as_query_engine(similarity_top_k=k)

    def __deepcopy__(self, memo):
        # return self instead of trying to copy internal state
        return self

    def dump_state(self, **kwargs):
        """Handle state dumping for DSPy serialization, accepting modern kwargs like json_mode."""
        return {"k": self.k}

    def load_state(self, state, **kwargs):
        """Handle state loading for DSPy serialization."""
        self.k = state.get("k", self.k)
        
    def forward(self, query_or_queries, k=None):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        k = k if k is not None else self.k
        
        all_passed_context = []
        for query in queries:
            response = self.query_engine.query(query)
            # Return a list of objects with long_text for DSPy compatibility
            contexts = [dspy.Prediction(long_text=n.node.get_content()) for n in response.source_nodes]
            all_passed_context.extend(contexts)
            
        return all_passed_context

def get_hdb_index():
    """Load or initialize the LlamaIndex for HDB chunks."""
    # Paths relative to project root
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "chunks.json"
    storage_dir = base_dir / "data" / "index_storage"
    
    if not storage_dir.exists():
        if not data_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {data_path}. Run parsing first.")
            
        with open(data_path, 'r') as f:
            chunks = json.load(f)
            
        print("Building Vector Index...")
        documents = [
            Document(
                text=c['text'], 
                metadata={"chunk_id": c['chunk_id'], "section": c['section']}
            ) for c in chunks
        ]
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=storage_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        
    return index
