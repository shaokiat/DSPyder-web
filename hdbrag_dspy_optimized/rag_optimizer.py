import dspy
import json
import random
import logging
from dspy import settings
from dotenv import load_dotenv
from src.retriever import get_hdb_index
from src.model import HDBRAG
from src.signatures import JudgeQA

# Configuration
QA_PAIRS_PATH = 'data/qa_pairs.json'
OPTIMIZED_RAG_PATH = "data/optimized_rag_qwen3:0.6b.json"
EVAL_RESULTS_PATH = "data/evaluation_results.json"
STUDENT_MODEL = 'ollama/qwen3:0.6b'
JUDGE_MODEL = 'ollama/qwen3:0.6b'
OLLAMA_API_BASE = 'http://localhost:11434'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dspy():
    """Initialize DSPy settings and models."""
    load_dotenv()
    
    student = dspy.LM(STUDENT_MODEL, api_base=OLLAMA_API_BASE)
    judge_lm = dspy.LM(JUDGE_MODEL, api_base=OLLAMA_API_BASE, cache=True, max_tokens=512, temperature=0)
    
    settings.configure(lm=student)
    return student, judge_lm

def load_data(file_path):
    """Load and format data for DSPy."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        example = dspy.Example(
            question=item['question'],
            answer=item['answer']
        ).with_inputs('question')
        examples.append(example)
    return examples

def get_metric(judge_lm):
    """Define the metric function using a judge model."""
    judge = dspy.Predict(JudgeQA)
    def metric(example, pred, trace=None):
        # Guard against bad predictions
        if not hasattr(pred, "answer"):
            return 0.0
        if pred.answer is None or pred.answer.strip() == "":
            return 0.0

        try:
            with dspy.settings.context(lm=judge_lm):
                result = judge(
                    question=example.question,
                    gold_answer=example.answer,
                    predicted_answer=pred.answer
                )
            # Coerce to float for Evaluate
            return float(result.is_accurate)
        except Exception as e:
            logger.error(f"Judge failure: {e}")
            return 0.0
    return metric

def save_evaluation_results(label, score, model_name, file_path):
    """Save evaluation results to a JSON file."""
    results = {}
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    results[label] = {
        "score": score,
        "model": model_name,
        "timestamp": dspy.settings.lm.history[-1]['timestamp'] if hasattr(dspy.settings.lm, 'history') and dspy.settings.lm.history else None
    }
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results for '{label}' saved to {file_path}")

def run_evaluation(rag_module, devset, metric, label):
    """Run evaluation and save results."""
    logger.info(f"Starting {label} evaluation...")
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=4,
        display_progress=True,
    )
    results = evaluator(rag_module)
    logger.info(f"{label.capitalize()} Evaluation Results: {results}")
    save_evaluation_results(label, float(results.score), STUDENT_MODEL, EVAL_RESULTS_PATH)
    return results

def main():
    # 1. Setup
    student_lm, judge_lm = setup_dspy()
    metric = get_metric(judge_lm)
    
    # 2. Load and Split Data
    all_examples = load_data(QA_PAIRS_PATH)
    random.seed(42)
    random.shuffle(all_examples)
    
    # Use 70/30 split
    split_idx = int(len(all_examples) * 0.7)
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]
    
    logger.info(f"Total examples loaded: {len(all_examples)}")
    logger.info(f"Trainset size: {len(train_examples)}")
    logger.info(f"Testset size: {len(test_examples)}")
    
    # 3. Initialize RAG
    index = get_hdb_index()
    rag = HDBRAG(index=index, k=3)
    
    # 4. Initial Evaluation
    run_evaluation(rag, test_examples, metric, "initial_pre_optimization")
    
    # 5. Optimization
    logger.info("Starting optimization with MIPROv2...")
    teleprompter = dspy.MIPROv2(
        metric=metric,
        auto='light',
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_threads=4
    )
    
    optimized_rag = teleprompter.compile(
        rag,
        trainset=train_examples,
    )
    
    optimized_rag.save(OPTIMIZED_RAG_PATH)
    logger.info(f"Optimized RAG saved to {OPTIMIZED_RAG_PATH}")
    
    # 6. Final Evaluation
    final_rag = HDBRAG(index=index, k=3)
    final_rag.load(OPTIMIZED_RAG_PATH)
    run_evaluation(final_rag, test_examples, metric, "final_post_optimization")

if __name__ == "__main__":
    main()