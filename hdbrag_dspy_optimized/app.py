import os
import argparse
import dspy
from pathlib import Path
from dotenv import load_dotenv
from src.retriever import get_hdb_index
from src.model import HDBRAG

def setup_model(model_name: str):
    """Setup DSPy LM based on model name."""
    if model_name == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment.")
        return dspy.LM("gpt-4o-mini")
    elif model_name == "ollama":
        return dspy.LM('ollama/qwen3:0.6b', api_base='http://localhost:11434')
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="HDB RAG Expert Chatbot")
    parser.add_argument("--model", type=str, default="ollama", choices=["openai", "ollama"], help="Model to use for chat")
    args = parser.parse_args()

    load_dotenv()

    # 3. Setup DSPy LM
    print(f"🤖 Initializing HDB RAG Expert with model: {args.model}...")
    try:
        lm = setup_model(args.model)
        dspy.settings.configure(lm=lm)
    except Exception as e:
        print(f"❌ Error setting up model {args.model}: {e}")
        return

    # 5. Initialize Knowledge Base
    try:
        index = get_hdb_index()
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        return

    # 6. Instantiate RAG Module
    rag = HDBRAG(index=index, k=3)

    print(f"\n✅ System Ready! Ask your questions using {args.model}.")
    print("(Type 'quit', 'exit', or 'q' to stop)\n")

    # 7. Chat Loop
    while True:
        try:
            query = input("👤 You: ").strip()
            
            if query.lower() in ["quit", "exit", "q", ""]:
                if not query: continue
                print("👋 Goodbye!")
                break

            print("🔍 Searching and thinking...")
            
            # Run the RAG program
            prediction = rag(question=query)

            # Display results
            print(f"\n🤖 Agent: {prediction.answer}")
            
            # Cited Contexts (Optional: show where it came from)
            if prediction.context:
                print("\n📚 Sources:")
                for i, ctx in enumerate(prediction.context):
                    # Show a snippet of the context for verification
                    snippet = ctx[:100].replace('\n', ' ') + "..."
                    print(f"   [{i+1}] {snippet}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Use 'quit' to exit.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
