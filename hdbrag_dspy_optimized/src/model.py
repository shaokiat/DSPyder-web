import dspy
from .signatures import GenerateAnswer, GenerateSearchQueries, GenerateHypotheticalAnswer
from .retriever import HDBRetriever

class HDBRAG(dspy.Module):
    """The core RAG module using Chain of Thought and HDB Retrieval."""
    def __init__(self, index, k=3):
        super().__init__()
        self.index = index
        self.k = k
        self.retriever = HDBRetriever(index=index, k=k)
        
        # Transformation layers
        self.generate_queries = dspy.Predict(GenerateSearchQueries)
        self.generate_hyde = dspy.Predict(GenerateHypotheticalAnswer)
        
        # Generator
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # 1. Multi-Query Expansion
        query_expansion = self.generate_queries(question=question).queries
        # Simple parsing if the model returns a string list
        queries = [question]
        if isinstance(query_expansion, list):
            queries.extend(query_expansion)
        elif isinstance(query_expansion, str):
            queries.extend([q.strip() for q in query_expansion.split("\n") if q.strip()][:2])

        # 2. HyDE (Hypothetical Document Embeddings)
        hyde_answer = self.generate_hyde(question=question).answer
        queries.append(hyde_answer)

        # 3. Enhanced Retrieval
        context = self.retriever(queries, k=self.k)
        
        # 4. Filter duplicates and generate
        seen_texts = set()
        unique_context = []
        for c in context:
            if c.long_text not in seen_texts:
                unique_context.append(c)
                seen_texts.add(c.long_text)

        prediction = self.generate_answer(context=unique_context[:self.k+2], question=question)
        return dspy.Prediction(context=unique_context, answer=prediction.answer)
