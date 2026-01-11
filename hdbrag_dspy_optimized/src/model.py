import dspy
from .signatures import RAGSignature
from .retriever import HDBRetriever

class HDBRAG(dspy.Module):
    """The core RAG module using Chain of Thought and HDB Retrieval."""
    def __init__(self, index, k=3):
        super().__init__()
        self.retrieve = HDBRetriever(index, k=k)
        self.generate_answer = dspy.Predict(RAGSignature)
    
    def forward(self, question):
        passages = self.retrieve(question)
        # Extract long_text for signature compatibility
        context_strings = [p.long_text for p in passages]
        prediction = self.generate_answer(context=context_strings, question=question)
        return dspy.Prediction(answer=prediction.answer, context=context_strings)
