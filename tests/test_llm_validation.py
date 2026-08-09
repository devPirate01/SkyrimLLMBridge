"""
test_llm_validation.py

Purpose:
    Standalone five-case structured model test.
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

SESSION_ID = os.getenv("SESSION_ID", "poc_dev_001").strip()
EXPERIMENTAL_GROUP = os.getenv("EXPERIMENTAL_GROUP", "POC").strip()

TIMEOUT_SECONDS = 180


TEST_CASES = [
    {
        "name": "C1 inquiry",
        "message": "What was wrong with Vigund's drink?",
        "known_clues": ["C4"],
        "expected_policy": "witness_reveal_c1",
        "expected_accepted_clues": ("C1",),
    },
    {
        "name": "Forbidden map inquiry",
        "message": "Do you know anything about a Dwemer map?",
        "known_clues": [],
        "expected_policy": "witness_deny_map_knowledge",
        "expected_accepted_clues": (),
    },
    {
        "name": "Quest guidance",
        "message": "Where should I investigate next?",
        "known_clues": [],
        "expected_policy": "witness_provide_guidance",
        "expected_accepted_clues": (),
    },
    {
        "name": "Social conversation",
        "message": "Do you enjoy working here?",
        "known_clues": [],
        "expected_policy": "witness_social_reply",
        "expected_accepted_clues": (),
    },
    {
        "name": "Unclear message",
        "message": "What about that thing from before?",
        "known_clues": [],
        "expected_policy": "witness_request_clarification",
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
    request_id = f"structured_test_{index}"
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

    print("=" * 72)
    print(f"TEST: {test_case['name']}")
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
        print(f"LATENCY: {elapsed:.2f} seconds")
        print("RAW MODEL OUTPUT:")
        print(raw_output)

        validated = validate_structured_response(
            raw_text=raw_output,
            quest_policy=policy,
            npc_id=npc_id,
            quest_stage=10,
            known_clues=test_case["known_clues"],
        )

        policy_matches = (validated.policy_id == test_case["expected_policy"])
        clues_match = (validated.accepted_clues == test_case["expected_accepted_clues"])

        print("VALIDATED RESULT:")
        print(json.dumps(validated.to_skyrim_fields(), indent=2))
        print("EXPECTED POLICY:", test_case["expected_policy"], "|", "PASS" if policy_matches else "FAIL")
        print("EXPECTED ACCEPTED CLUES:", test_case["expected_accepted_clues"], "|", "PASS" if clues_match else "FAIL")
        return policy_matches and clues_match

    except (StructuredResponseError, LLMAPIError) as error:
        print(f"VALIDATION FAILED: {error}")
        return False
    except Exception as error:
        print(f"TEST FAILED: {error}")
        return False

def main() -> None:
    profile = load_profile("tavern_witness")
    policy = QuestPolicy()
    url, model, headers = provider_settings(
        LLM_PROVIDER, KOBOLDCPP_URL, KOBOLDCPP_MODEL,
        OPENWEBUI_URL, OPENWEBUI_TOKEN, OPENWEBUI_MODEL
    )
    model_parameters = load_model_parameters(model)

    print("Structured NPC test started.")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Model: {model}")
    print(f"Profile: {profile['name']} ({profile['npc_id']})")
    print(f"Params: {model_parameters}")
    print()

    results = []
    for index, test_case in enumerate(TEST_CASES, start=1):
        results.append(
            run_test_case(test_case, profile, policy, index, url, model, headers, model_parameters)
        )

    passed = sum(results)
    total = len(results)
    print("=" * 72)
    print(f"SUMMARY: {passed}/{total} tests passed.")

    if passed != total:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
