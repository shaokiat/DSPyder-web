import json
from pathlib import Path
import dspy
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank

class HDBRetriever(dspy.Retrieve):
    """
    A custom DSPy retriever that uses LlamaIndex for Hybrid Search (Vector + BM25)
    and Reranking (Cross-Encoder).
    """
    def __init__(self, index, k=3):
        super().__init__(k=k)
        self.index = index
        
        # 1. Setup Vector Retriever
        vector_retriever = index.as_retriever(similarity_top_k=k * 2)
        
        # 2. Setup BM25 Retriever
        nodes = list(index.docstore.docs.values())
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k * 2)
        
        # 3. Setup Hybrid Search (Query Fusion)
        self.hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=k * 3, # Get more candidates for reranking
            num_queries=1, # Default to 1, can expansion later
            mode="reciprocal_rerank",
            use_async=False
        )
        
        # 4. Setup Reranker
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-TinyBERT-L-2-v2", 
            top_n=k
        )

    def __deepcopy__(self, memo):
        return self

    def dump_state(self, **kwargs):
        return {"k": self.k}

    def load_state(self, state, **kwargs):
        self.k = state.get("k", self.k)
        
    def forward(self, query_or_queries, k=None):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        k = k if k is not None else self.k
        
        all_passed_context = []
        for query in queries:
            # First pass: Hybrid retrieval
            nodes = self.hybrid_retriever.retrieve(query)
            
            # Second pass: Reranking
            from llama_index.core.schema import QueryBundle
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))
            
            # Return top k
            contexts = [dspy.Prediction(long_text=n.node.get_content()) for n in reranked_nodes[:k]]
            all_passed_context.extend(contexts)
            
        return all_passed_context

def get_hdb_index(force_rebuild=False):
    """Load or initialize the LlamaIndex for HDB chunks."""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "chunks.json"
    storage_dir = base_dir / "data" / "index_storage"
    
    if force_rebuild or not storage_dir.exists():
        if not data_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {data_path}. Run parsing first.")
            
        with open(data_path, 'r') as f:
            chunks = json.load(f)
            
        print("Building Hybrid Vector Index...")
        documents = [
            Document(
                text=c['text'], 
                metadata={
                    "chunk_id": c['chunk_id'], 
                    "section": c['section'],
                    "doc_id": c['doc_id']
                }
            ) for c in chunks
        ]
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=storage_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        
    return index
