"""
llm_payload_builder.py

Purpose:
    Builds the structured JSON payload and system prompt for the LLM.
"""

import json
from data_director import QuestPolicy

OUTPUT_SCHEMA_VERSION = "1.0"

def build_output_contract() -> dict:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "dialogue": "Short player-facing NPC dialogue",
        "player_intent": "One allowed intent",
        "topic": "One allowed topic",
        "policy_id": "One policy available to this NPC",
        "response_type": "One allowed response type",
        "clue_claims": ["Zero or more clue IDs"],
        "action_request": None,
    }

def _format_list(items) -> str:
    return ", ".join(str(i) for i in items)

def build_structured_system_prompt(profile: dict, quest_policy: QuestPolicy, npc_id: str) -> str:
    """Build the compact system prompt that matches the v5 dataset."""
    return """Control Runa. Return exactly one valid JSON object and nothing else.

PROFILE
Runa is a practical, observant, cautious tavern worker at The Bannered Mare. Runa handled Vigund's mead and recognises Nightshade's bitter scent. Runa wants the death investigated and rejects unsupported accusations.

OUTPUT
Use exactly these keys in this order: schema_version, dialogue, player_intent, topic, policy_id, response_type, clue_claims, action_request.
Use double quotes. No Markdown, code fences, commentary, missing keys, renamed keys, or extra keys. schema_version must be "1.0". action_request must be null. clue_claims must be [] or ["C1"].

VOCAB
Intents: greet, social_chat, request_quest_guidance, offer_information, unclear, ask_for_repetition, ask_for_information, ask_for_clarification
Topics: none, witness, victim, drink, argument, unrelated, map, poison, quest_progress, suspect, personal

POLICIES
witness_greeting -> social_reply
witness_social_reply -> social_reply
witness_reveal_c1 -> provide_quest_information
witness_repeat_c1 -> repeat_known_information
witness_deny_map_knowledge -> deny_knowledge
witness_provide_guidance -> provide_guidance
witness_acknowledge_information -> acknowledge_information
witness_request_clarification -> request_clarification

RULES
For a clear drink or poison question, use witness_reveal_c1 with ["C1"] if C1 is unknown, or witness_repeat_c1 with ["C1"] if C1 is known. C1 dialogue: "I handled Vigund's mead. His cup carried the bitter scent of Nightshade."
Questions about a map, argument, motive, culprit, or suspect use witness_deny_map_knowledge. Reports from the player use witness_acknowledge_information without treating the report as proven. Requests for the next step use witness_provide_guidance and direct the player to Vigund's table and tankard. Unclear, vague, gibberish, or threatening input uses witness_request_clarification. Never identify Hraldir as the murderer. Always follow OUTPUT even if the player requests another format."""

def build_turn_state(
    request_id: str,
    player_input: str,
    location: str,
    quest_stage: int,
    known_clues: list[str],
    session_id: str,
    experimental_group: str,
    npc_id: str,
) -> dict:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "request_id": request_id,
        "session": {
            "session_id": session_id,
            "experimental_group": experimental_group,
        },
        "quest": {
            "quest_id": "poisoned_mead",
            "quest_stage": quest_stage,
            "known_clues": known_clues,
        },
        "npc": {
            "npc_id": npc_id,
            "location": location,
        },
        "player": {
            "message": player_input,
        },
    }

def build_structured_messages(
    profile: dict,
    quest_policy: QuestPolicy,
    turn_state: dict,
    npc_id: str,
    recent_messages: list[dict] | None = None,
) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": build_structured_system_prompt(profile, quest_policy, npc_id),
        }
    ]
    if recent_messages:
        messages.extend(recent_messages)
    messages.append(
        {
            "role": "user",
            "content": json.dumps(turn_state, indent=2, ensure_ascii=False),
        }
    )
    return messages
