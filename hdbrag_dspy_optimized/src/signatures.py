import dspy

class RAGSignature(dspy.Signature):
    """Answer the question based on the provided HDB context.
    The answer must be grounded ONLY in the context. If the information is missing, say you don't know.
    """
    context = dspy.InputField(desc="Relevant HDB documentation snippets")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A factual answer based on the context. Plain text answer. Do not return JSON or structured data.")

class GenerateRAGUsageExample(dspy.Signature):
    """Generate a realistic user query and a grounded answer based on the provided context.
    The query should simulate how a real user might ask for information about HDB BTO/grants.
    The answer must be factual and grounded ONLY in the provided context.
    """
    context = dspy.InputField(desc="A snippet of HDB documentation")
    user_query = dspy.OutputField(desc="A realistic user question or request")
    grounded_answer = dspy.OutputField(desc="A factual answer based strictly on the context")

class JudgeQA(dspy.Signature):
    """
    Decide whether the predicted answer is correct based on the gold answer.

    Evaluation Criteria:
    - Correct if: It states the same fact as the gold answer, OR both indicate missing/unknown information.
    - Incorrect if: It contradicts the gold answer or invents information (hallucination).
    """
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()

    is_accurate: bool = dspy.OutputField(
        desc="True if the predicted answer is semantically equivalent to the gold answer"
    )
