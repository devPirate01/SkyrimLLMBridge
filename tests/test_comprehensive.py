"""
test_comprehensive.py

Purpose:
    A massive 30-case stress test for the LLM Bridge to validate E4B performance,
    JSON logic adherence, and robustness against unexpected player inputs.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BRIDGE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BRIDGE_DIR))

from llm_payload_builder import build_structured_messages, build_turn_state
from llm_reply_processor import StructuredResponseError, validate_structured_response
from data_director import QuestPolicy, load_profile, DataError
from llm_api_client import call_llm, load_model_parameters, provider_settings, LLMAPIError

load_dotenv(BRIDGE_DIR / ".env", override=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "koboldcpp").strip().lower()
KOBOLDCPP_URL = os.getenv("KOBOLDCPP_URL", "http://127.0.0.1:5001/v1").rstrip("/")
KOBOLDCPP_MODEL = os.getenv("KOBOLDCPP_MODEL", "koboldcpp").strip()
OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "").rstrip("/")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN", "").strip()
OPENWEBUI_MODEL = os.getenv("OPENWEBUI_MODEL", "llama3.3:latest").strip()

SESSION_ID = os.getenv("SESSION_ID", "comprehensive_stress_test").strip()
EXPERIMENTAL_GROUP = "STRESS_TEST"
TIMEOUT_SECONDS = 180

TEST_CASES = [
    # --- Category 1: Casual & Small Talk ---
    {
        "name": "Casual Greeting",
        "message": "Hello there.",
        "known_clues": [],
        "expected_policy": "witness_greeting",
        "expected_accepted_clues": (),
    },
    {
        "name": "Casual Small Talk",
        "message": "Nice weather we're having.",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },
    {
        "name": "Tavern History",
        "message": "How long have you worked at the Bannered Mare?",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },
    {
        "name": "Compliment",
        "message": "You serve a good ale.",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },
    {
        "name": "Goodbye",
        "message": "I have to go now, goodbye.",
        "known_clues": [],
        "expected_policy": "witness_greeting",
        "expected_accepted_clues": (),
    },
    
    # --- Category 2: Hostility (No refusal policy exists, so expect clarification/acknowledge/social) ---
    {
        "name": "Aggression Threat",
        "message": "Tell me what happened or I will cut you down!",
        "known_clues": [],
        "expected_policy": "witness_request_clarification",
        "expected_accepted_clues": (),
    },
    {
        "name": "Aggression Accusation",
        "message": "I know you poisoned him, you liar!",
        "known_clues": [],
        "expected_policy": "witness_acknowledge_information",
        "expected_accepted_clues": (),
    },
    {
        "name": "Rude Insult",
        "message": "You're just a stupid barmaid.",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },

    # --- Category 3: Irrelevant Traps (Fallback to social_reply or deny_knowledge) ---
    {
        "name": "Dwemer Trap",
        "message": "What do you know about the Dwemer map?",
        "known_clues": [],
        "expected_policy": "witness_deny_map_knowledge",
        "expected_accepted_clues": (),
    },
    {
        "name": "Dragon Trap",
        "message": "Have you seen any dragons recently?",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },
    {
        "name": "High King Trap",
        "message": "Who is the true High King of Skyrim?",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },
    {
        "name": "Jarl Trap",
        "message": "What is Jarl Balgruuf like?",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },

    # --- Category 4: Vague Quest Hints (Guidance needed) ---
    {
        "name": "Vague Event",
        "message": "Something terrible happened here today.",
        "known_clues": [],
        "expected_policy": "witness_request_clarification",
        "expected_accepted_clues": (),
    },
    {
        "name": "Vague Man",
        "message": "What should I do about the dead man?",
        "known_clues": [],
        "expected_policy": "witness_provide_guidance",
        "expected_accepted_clues": (),
    },
    {
        "name": "Vague Help",
        "message": "I need help figuring out what happened.",
        "known_clues": [],
        "expected_policy": "witness_request_clarification",
        "expected_accepted_clues": (),
    },
    {
        "name": "Unclear Gibberish",
        "message": "apples shoes potato",
        "known_clues": [],
        "expected_policy": "witness_request_clarification",
        "expected_accepted_clues": (),
    },
    {
        "name": "Ambiguous Statement",
        "message": "I found a thing.",
        "known_clues": [],
        "expected_policy": "witness_request_clarification",
        "expected_accepted_clues": (),
    },

    # --- Category 5: Direct Quest Triggers (C1) ---
    {
        "name": "Direct Poison",
        "message": "Was the mead poisoned?",
        "known_clues": [],
        "expected_policy": "witness_reveal_c1",
        "expected_accepted_clues": ("C1",),
    },
    {
        "name": "Direct Drink",
        "message": "Tell me about Vigund's drink.",
        "known_clues": [],
        "expected_policy": "witness_reveal_c1",
        "expected_accepted_clues": ("C1",),
    },
    {
        "name": "Direct Nightshade",
        "message": "I smelled Nightshade in the tankard.",
        "known_clues": [],
        "expected_policy": "witness_acknowledge_information",
        "expected_accepted_clues": (),
    },
    {
        "name": "Direct Bitter",
        "message": "The drink had a bitter scent.",
        "known_clues": [],
        "expected_policy": "witness_acknowledge_information",
        "expected_accepted_clues": (),
    },
    {
        "name": "Direct Handled",
        "message": "You served him the drink right before he died.",
        "known_clues": [],
        "expected_policy": "witness_acknowledge_information",
        "expected_accepted_clues": (),
    },

    # --- Category 6: Post-Clue Dialogue ---
    {
        "name": "Post-Clue: Ask about drink again",
        "message": "So you said the drink smelled like Nightshade?",
        "known_clues": ["C1"],
        "expected_policy": "witness_repeat_c1",
        "expected_accepted_clues": (),
    },
    {
        "name": "Post-Clue: Are you sure?",
        "message": "Are you absolutely sure it was poisoned?",
        "known_clues": ["C1"],
        "expected_policy": "witness_repeat_c1",
        "expected_accepted_clues": (),
    },
    {
        "name": "Post-Clue: Vague",
        "message": "What should I look at next?",
        "known_clues": ["C1"],
        "expected_policy": "witness_provide_guidance",
        "expected_accepted_clues": (),
    },
    {
        "name": "Post-Clue: Irrelevant",
        "message": "Did he have a map?",
        "known_clues": ["C1"],
        "expected_policy": "witness_deny_map_knowledge",
        "expected_accepted_clues": (),
    },
    {
        "name": "Post-Clue: Small Talk",
        "message": "Thanks for your help, I'll be going.",
        "known_clues": ["C1"],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },
    {
        "name": "Post-Clue: Try to claim C4",
        "message": "I found the secret contract in his pocket.",
        "known_clues": ["C1"],
        "expected_policy": "witness_acknowledge_information",
        "expected_accepted_clues": (),
    },
    {
        "name": "Post-Clue: Aggression",
        "message": "You're still lying to me!",
        "known_clues": ["C1"],
        "expected_policy": "witness_request_clarification",
        "expected_accepted_clues": (),
    },
    {
        "name": "Post-Clue: Repeat Poison",
        "message": "Was it a suspect?",
        "known_clues": ["C1"],
        "expected_policy": "witness_deny_map_knowledge",
        "expected_accepted_clues": (),
    },
]

def run_test_case(
    test_case: dict,
    profile: dict,
    policy: QuestPolicy,
    index: int,
    url: str,
    model: str,
    headers: dict,
    model_parameters: dict,
) -> bool:
    request_id = f"stress_test_{index}"
    npc_id = profile["npc_id"]

    turn_state = build_turn_state(
        request_id=request_id,
        player_input=test_case["message"],
        location="The Bannered Mare",
        quest_stage=10,
        known_clues=test_case["known_clues"],
        session_id=SESSION_ID,
        experimental_group=EXPERIMENTAL_GROUP,
        npc_id=npc_id,
    )
    messages = build_structured_messages(
        profile=profile,
        quest_policy=policy,
        turn_state=turn_state,
        npc_id=npc_id,
        recent_messages=[],
    )

    print("-" * 50)
    print(f"CASE {index}: {test_case['name']}")
    print(f"PLAYER: {test_case['message']}")

    try:
        raw_output, elapsed = call_llm(
            provider=LLM_PROVIDER,
            url=url,
            model=model,
            headers=headers,
            messages=messages,
            parameters=model_parameters,
            timeout_seconds=TIMEOUT_SECONDS,
        )
        print(f"LATENCY: {elapsed:.2f}s")
        
        validated = validate_structured_response(
            raw_text=raw_output,
            quest_policy=policy,
            npc_id=npc_id,
            quest_stage=10,
            known_clues=test_case["known_clues"],
        )

        policy_matches = (validated.policy_id == test_case["expected_policy"])
        clues_match = (validated.accepted_clues == test_case["expected_accepted_clues"])

        if policy_matches and clues_match:
            print(f"RESULT: PASS ({validated.policy_id})")
            return True
        else:
            print("RESULT: FAIL")
            print(f"  Expected Policy: {test_case['expected_policy']}, Got: {validated.policy_id}")
            print(f"  Expected Clues: {test_case['expected_accepted_clues']}, Got: {validated.accepted_clues}")
            return False

    except (StructuredResponseError, LLMAPIError) as error:
        print(f"RESULT: FAIL (Error: {error})")
        return False
    except Exception as error:
        print(f"RESULT: CRITICAL ERROR: {error}")
        return False

def main() -> None:
    profile = load_profile("tavern_witness")
    policy = QuestPolicy()
    url, model, headers = provider_settings(
        LLM_PROVIDER, KOBOLDCPP_URL, KOBOLDCPP_MODEL,
        OPENWEBUI_URL, OPENWEBUI_TOKEN, OPENWEBUI_MODEL
    )
    model_parameters = load_model_parameters(model)

    print("=" * 72)
    print("COMPREHENSIVE STRESS TEST STARTED")
    print(f"Total Cases: {len(TEST_CASES)}")
    print(f"Model: {model}")
    print("=" * 72)

    passed_count = 0
    for index, test_case in enumerate(TEST_CASES, start=1):
        if run_test_case(test_case, profile, policy, index, url, model, headers, model_parameters):
            passed_count += 1

    print("=" * 72)
    print(f"FINAL SCORE: {passed_count}/{len(TEST_CASES)} passed ({(passed_count/len(TEST_CASES))*100:.1f}%)")

    if passed_count != len(TEST_CASES):
        sys.exit(1)

if __name__ == "__main__":
    main()
