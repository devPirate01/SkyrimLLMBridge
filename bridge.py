"""
bridge.py

Purpose:
    The main execution loop for the Skyrim LLM Bridge.
    Orchestrates file I/O, API calls, and state management.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from data_director import (
    DataError,
    QuestPolicy,
    append_log,
    load_profile,
    save_exchange,
    write_json_atomic,
)
from llm_api_client import LLMAPIError, call_llm, load_model_parameters, provider_settings
from llm_payload_builder import build_structured_messages, build_turn_state
from llm_reply_processor import StructuredResponseError, validate_structured_response


BRIDGE_DIRECTORY = Path(__file__).resolve().parent
load_dotenv(BRIDGE_DIRECTORY / ".env", override=True)

SKYRIM_DATA_DIR = Path(r"H:\Games\TES - Skyrim - Anniversary Edition\Data\SKSE\Plugins\StorageUtilData\CompanionLLM")
REQUEST_PATH = SKYRIM_DATA_DIR / "request.json"
RESPONSE_PATH = SKYRIM_DATA_DIR / "response.json"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "koboldcpp").strip().lower()
KOBOLDCPP_URL = os.getenv("KOBOLDCPP_URL", "http://127.0.0.1:5001/v1").rstrip("/")
KOBOLDCPP_MODEL = os.getenv("KOBOLDCPP_MODEL", "koboldcpp").strip()
OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "").rstrip("/")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN", "").strip()
OPENWEBUI_MODEL = os.getenv("OPENWEBUI_MODEL", "llama3.3:latest").strip()

SESSION_ID = os.getenv("SESSION_ID", "default_session").strip()
PARTICIPANT_ID = os.getenv("PARTICIPANT_ID", "default_participant").strip()
EXPERIMENTAL_GROUP = os.getenv("EXPERIMENTAL_GROUP", "control").strip()
DIALOGUE_CONDITION = os.getenv("DIALOGUE_CONDITION", "structured").strip()

POLL_INTERVAL_SECONDS = 0.1
TIMEOUT_SECONDS = 180

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def process_request(
    quest_policy: QuestPolicy,
    url: str,
    model: str,
    headers: dict,
    model_parameters: dict,
    current_content: str,
) -> None:
    request_data = json.loads(current_content)
    string_data = request_data.get("string", {})
    int_data = request_data.get("int", {})
    
    received_utc = utc_now()
    request_id = string_data.get("request_id", "unknown")
    npc_name = string_data.get("npc_name", "tavern_witness")
    location = string_data.get("location", "unknown")
    player_input = string_data.get("player_input", "")
    quest_stage = int(int_data.get("quest_stage", 0))
    
    known_clues = []
    if int(int_data.get("c1_acquired", 0)) == 1:
        known_clues.append("C1")
    if int(int_data.get("c4_acquired", 0)) == 1:
        known_clues.append("C4")

    print(f"\n[{received_utc}] Received request: {request_id}")
    print(f"Player: {player_input}")

    # Dynamically load the correct NPC profile based on the request
    npc_profile = load_profile(npc_name)
    npc_id = npc_profile["npc_id"]

    turn_state = build_turn_state(
        request_id=request_id,
        player_input=player_input,
        location=location,
        quest_stage=quest_stage,
        known_clues=known_clues,
        session_id=SESSION_ID,
        experimental_group=EXPERIMENTAL_GROUP,
        npc_id=npc_id,
    )

    from data_director import load_memory
    recent_messages = load_memory(npc_name, SESSION_ID)

    messages = build_structured_messages(
        profile=npc_profile,
        quest_policy=quest_policy,
        turn_state=turn_state,
        npc_id=npc_id,
        recent_messages=recent_messages,
    )

    started_utc = utc_now()
    raw_output, latency = call_llm(
        provider=LLM_PROVIDER,
        url=url,
        model=model,
        headers=headers,
        messages=messages,
        parameters=model_parameters,
        timeout_seconds=TIMEOUT_SECONDS,
    )
    finished_utc = utc_now()
    print(f"Model replied in {latency:.2f} seconds.")

    validated = validate_structured_response(
        raw_text=raw_output,
        quest_policy=quest_policy,
        npc_id=npc_id,
        quest_stage=quest_stage,
        known_clues=known_clues,
    )

    skyrim_fields = validated.to_skyrim_fields()
    skyrim_fields["request_id"] = request_id
    skyrim_fields["npc_name"] = npc_name
    write_json_atomic(RESPONSE_PATH, {"string": skyrim_fields})
    response_written_utc = utc_now()
    print(f"Accepted response: {validated.dialogue}")

    save_exchange(npc_name, SESSION_ID, player_input, validated.dialogue)
    append_log(SESSION_ID, {
        "session_id": SESSION_ID,
        "participant_id": PARTICIPANT_ID,
        "experimental_group": EXPERIMENTAL_GROUP,
        "dialogue_condition": DIALOGUE_CONDITION,
        "request_id": request_id,
        "npc_name": npc_name,
        "location": location,
        "provider": LLM_PROVIDER,
        "model": model,
        "bridge_received_utc": received_utc,
        "model_started_utc": started_utc,
        "model_finished_utc": finished_utc,
        "response_written_utc": response_written_utc,
        "model_latency_ms": int(latency * 1000),
        "player_input": player_input,
        "response_text": validated.dialogue,
        "quest_stage": quest_stage,
        "claimed_clues": ",".join(validated.clue_claims),
        "accepted_clues": ",".join(validated.accepted_clues),
    })

def main() -> None:
    print("Initialising Skyrim LLM Bridge (Dynamic NPCs)...")
    try:
        quest_policy = QuestPolicy()
        url, model, headers = provider_settings(
            LLM_PROVIDER, KOBOLDCPP_URL, KOBOLDCPP_MODEL,
            OPENWEBUI_URL, OPENWEBUI_TOKEN, OPENWEBUI_MODEL
        )
        model_parameters = load_model_parameters(model)
    except (DataError, LLMAPIError) as error:
        sys.exit(f"Startup error: {error}")

    print(f"Provider: {LLM_PROVIDER}")
    print(f"Model: {model}")
    print(f"Params: {model_parameters}")
    print(f"Session ID: {SESSION_ID}")
    print("Waiting for request.json...\n")

    last_seen_content = None
    if REQUEST_PATH.exists():
        try:
            last_seen_content = REQUEST_PATH.read_text(encoding="utf-8")
            print("Existing request.json ignored. Waiting for a new request.")
        except OSError as error:
            print(f"Could not read existing request.json: {error}")

    while True:
        try:
            if REQUEST_PATH.exists():
                current_content = REQUEST_PATH.read_text(encoding="utf-8")
                if current_content != last_seen_content:
                    last_seen_content = current_content
                    process_request(quest_policy, url, model, headers, model_parameters, current_content)
        except Exception as error:
            print(f"Error processing request: {error}")
            write_json_atomic(RESPONSE_PATH, {"string": {"response": "I cannot speak right now.", "action": "", "request_id": "error", "npc_name": "error"}})
            append_log(SESSION_ID, {"error": str(error)})
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
