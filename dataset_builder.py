import json
import random
from pathlib import Path
import sys

# Import the actual bridge logic to guarantee 1:1 inference matching
sys.path.append(str(Path(__file__).resolve().parent))
from data_director import load_profile, QuestPolicy
from llm_payload_builder import build_structured_system_prompt, build_turn_state

def build_authentic_dataset(external_data_file=None, num_samples=300):
    profile = load_profile("tavern_witness")
    policy = QuestPolicy()
    npc_id = profile["npc_id"]
    
    # Generate the EXACT, massive system prompt used in production
    system_prompt_content = build_structured_system_prompt(profile, policy, npc_id)
    
    dataset = []
    
    # Guard: this builder requires the external data file from Copilot/Gemini.
    # Do NOT allow silent fallback to stale base_scenarios with broken vocabulary.
    if external_data_file is None or not Path(external_data_file).exists():
        raise FileNotFoundError(
            f"External dataset file required but not found: {external_data_file}\n"
            "Please provide a valid 'perfect_dataset_cleaned.json' or use dataset_train_v3.jsonl directly."
        )
    
    with open(external_data_file, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
        
    for i, scenario in enumerate(scenarios):
        request_id = f"train_req_{i}"
        
        # The other LLM should provide the known_clues for the scenario
        known_clues = scenario.get("known_clues", [])
        
        # Build the exact turn_state JSON payload the game sends
        turn_state = build_turn_state(
            request_id=request_id,
            player_input=scenario["player_input"],
            location="The Bannered Mare",
            quest_stage=10,
            known_clues=known_clues,
            session_id="train_session",
            experimental_group="train",
            npc_id=npc_id
        )
        
        # We now trust the external LLM to have generated the exact target JSON 
        # (including the dialogue and strict vocabulary).
        target_json = scenario["target_json"]
        
        sample = {
            "messages": [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": json.dumps(turn_state, indent=2, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(target_json)}
            ]
        }
        dataset.append(sample)
        
    # Split into 90% train, 10% eval
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]
    
    return train_data, eval_data

if __name__ == "__main__":
    external_file = Path(__file__).parent / "perfect_dataset_cleaned.json" 
    
    train, eval_set = build_authentic_dataset(external_file, 500)
    
    with open("dataset_train_v3.jsonl", "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
            
    with open("dataset_eval_v3.jsonl", "w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(train)} training and {len(eval_set)} eval samples.")
