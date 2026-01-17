import json
import random
import re
from collections import defaultdict
from pathlib import Path

def normalize_text(text):
    if text is None: return ""
    return re.sub(r'\W+', ' ', text.lower()).strip()

def get_similarity_key(question):
    words = normalize_text(question).split()
    if not words: return ""
    return " ".join(sorted(words[:6]))

def save_splits(input_path="data/qa_pairs.json", output_path="data/qa_split.json"):
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    # 0. Filter invalid data (same threshold as before)
    data = []
    for item in raw_data:
        q = item.get("question")
        a = item.get("answer")
        if q and isinstance(q, str) and len(q.strip()) > 10 and a and isinstance(a, str) and len(a.strip()) > 10:
            data.append(item)
    
    # 1. Infer topics and group by similarity
    sim_groups = defaultdict(list)
    topic_counts = defaultdict(int)
    
    for item in data:
        topic = item.get("section", "General")
        if topic == "General" or not topic:
             topic = item.get("doc_id", "General").replace("_", " ").title()
        item["inferred_topic"] = topic
        topic_counts[topic] += 1
        
        sim_key = get_similarity_key(item["question"])
        sim_groups[sim_key].append(item)

    # 2. Split strategy (balancing representation)
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1])
    under_represented = set([t[0] for t in sorted_topics[:len(sorted_topics)//2]])
    
    groups = list(sim_groups.values())
    random.seed(42)
    random.shuffle(groups)
    
    train, dev, test = [], [], []
    
    # First pass: dev and test (prioritizing rare topics)
    remaining_groups = []
    for group in groups:
        is_under = any(item["inferred_topic"] in under_represented for item in group)
        if is_under:
            if len(dev) < 30:
                dev.extend(group)
            elif len(test) < 30:
                test.extend(group)
            else:
                remaining_groups.append(group)
        else:
            remaining_groups.append(group)
            
    # Second pass: fill remaining
    for group in remaining_groups:
        if len(train) < 40:
            train.extend(group)
        elif len(dev) < 30:
            dev.extend(group)
        elif len(test) < 30:
            test.extend(group)

    # Final trim
    result = {
        "train": train[:40],
        "dev": dev[:30],
        "test": test[:30]
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Split saved successfully to {output_path}")
    print(f"Sizes: Train={len(result['train'])}, Dev={len(result['dev'])}, Test={len(result['test'])}")

if __name__ == "__main__":
    save_splits()
