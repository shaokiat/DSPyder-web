import json
import random
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import dspy
from src.signatures import GenerateRAGUsageExample

# Load environment variables
load_dotenv()

def generate_usage_examples(num_examples=10, output_file="data/qa_pairs.json"):
    # Setup DSPy

    # lm = dspy.LM(
    #     'openai/gpt-4o-mini', 
    #     cache=True,
    #     max_tokens=512,
    #     temperature=0.7 # Slight temperature for more diverse realistic queries
    # )
    lm = dspy.LM('ollama/qwen3:0.6b', api_base='http://localhost:11434', cache=True, max_tokens=512, temperature=0.2)

    dspy.settings.configure(lm=lm)
    
    # Locate data directory relative to project root
    # This script is in src/ingestion/, so go up 3 levels to get to project root
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent.parent
    
    # If run from root, current_dir might be different, but using __file__ is safer
    # Assuming standard structure:
    # PROJECT_ROOT/
    #   data/
    #   src/
    #     ingestion/
    #       qa_generator.py
    
    chunks_path = project_root / "data" / "chunks.json"
    
    if not chunks_path.exists():
        # Fallback: check relative to CWD if running from root
        cwd_chunks = Path("data/chunks.json")
        if cwd_chunks.exists():
             chunks_path = cwd_chunks
        else:
            print(f"Error: {chunks_path} not found.")
            return

    print(f"Loading chunks from: {chunks_path}")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    
    # Filter for valid chunks (has text and reasonable length)
    valid_chunks = [c for c in chunks if c.get("text") and len(c.get("text", "")) > 100]
    
    if not valid_chunks:
        print("Error: No valid chunks found.")
        return

    # Randomly sample chunks
    sampled_chunks = random.sample(valid_chunks, min(num_examples, len(valid_chunks)))
    
    generator = dspy.ChainOfThought(GenerateRAGUsageExample)
    
    results = []
    print(f"Generating {len(sampled_chunks)} usage examples...")
    
    for i, chunk in enumerate(sampled_chunks):
        context = chunk["text"]
        try:
            # Generate prediction using DSPy
            prediction = generator(context=context)
            
            example = {
                "doc_id": chunk.get("doc_id", "unknown"),
                "section": chunk.get("section", "unknown"),
                "question": prediction.user_query,
                "answer": prediction.grounded_answer,
                "context": context
            }
            results.append(example)
            print(f"[{i+1}] Generated: {prediction.user_query[:50]}...")
            
        except Exception as e:
            print(f"[{i+1}] Error generating example: {e}")
            
    # Save results
    # Handle output path
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = project_root / output_file
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSuccessfully generated {len(results)} examples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic RAG usage examples from HDB chunks using DSPy.")
    parser.add_argument("--num", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/qa_pairs.json", help="Output file path (relative to project root)")
    
    args = parser.parse_args()
    generate_usage_examples(num_examples=args.num, output_file=args.output)
