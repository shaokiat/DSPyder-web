import dspy

class GenerateAnswer(dspy.Signature):
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

class GenerateSearchQueries(dspy.Signature):
    """Generate multiple search queries to find information for the given question."""
    question = dspy.InputField()
    queries = dspy.OutputField(desc="List of search queries to broaden discovery")

class GenerateHypotheticalAnswer(dspy.Signature):
    """Generate a short, hypothetical answer to the question to help find relevant documents."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A brief hypothetical answer with key terms")

class JudgeQA(dspy.Signature):
    """Evaluate if the predicted answer accurately reflects the gold answer for the given question."""
    question = dspy.InputField()
    gold_answer = dspy.InputField()
    predicted_answer = dspy.OutputField()
    is_accurate = dspy.OutputField(desc="Boolean: True if the answer is factually correct, False otherwise")
