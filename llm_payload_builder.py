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
    """Build a heavily compressed, token-efficient system prompt."""
    
    profile_text = (
        f"Name: {profile.get('name', 'Unknown')}\n"
        f"Role: {profile.get('role', 'Unknown')}\n"
        f"Personality: {_format_list(profile.get('personality', []))}\n"
        f"Background: {_format_list(profile.get('background', []))}\n"
        f"Behaviour Rules: {_format_list(profile.get('behaviour_rules', profile.get('response_rules', [])))}\n"
    )

    vocab_text = (
        f"Allowed Intents: {_format_list(quest_policy.allowed_intents)}\n"
        f"Allowed Topics: {_format_list(quest_policy.allowed_topics)}\n"
        f"Allowed Response Types: {_format_list(quest_policy.allowed_response_types)}\n"
        f"Allowed Clues: {_format_list(quest_policy.allowed_clues_for_npc(npc_id))}\n"
        f"Forbidden Clues: {_format_list(quest_policy.forbidden_clues_for_npc(npc_id))}\n"
    )

    npc_policies = quest_policy.policies_for_npc(npc_id)
    policies_text = ""
    for p in npc_policies:
        policies_text += f"- Policy ID: {p['policy_id']}\n"
        if "when" in p:
            policies_text += f"  When Intent in [{_format_list(p['when'].get('intents',[]))}] AND Topic in [{_format_list(p['when'].get('topics',[]))}]\n"
        if "response_directive" in p:
            policies_text += f"  Goal: {p['response_directive'].get('goal', '')}\n"
        if p.get("award_clues"):
            policies_text += f"  Awards Clues: [{_format_list(p['award_clues'])}]\n"
        if "response_type" in p:
            policies_text += f"  Required Response Type: {p['response_type']}\n"

    return f"""You are controlling one Skyrim NPC conversation turn.

Interpret the player's communicative intent and topic, select exactly one
eligible dialogue policy, and generate the NPC's in-character reply.

Return only one valid JSON object. Do not use Markdown or code fences.
Do not add text before or after the JSON object.

CHARACTER PROFILE
{profile_text}
CONTROLLED VOCABULARY
{vocab_text}
ELIGIBLE QUEST POLICIES
{policies_text}
REQUIRED OUTPUT SHAPE
{json.dumps(build_output_contract(), indent=2)}

RULES
- Use only values from the controlled vocabulary.
- Follow the selected policy's response directive.
- Your response_type MUST exactly match the Required Response Type for your selected policy.
- Keep dialogue to no more than three short sentences.
- clue_claims may contain only clues actually communicated in dialogue.
- Never return a forbidden clue.
- action_request must be null for this proof of concept.
- If the player's meaning is unclear, select witness_request_clarification.
"""

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
