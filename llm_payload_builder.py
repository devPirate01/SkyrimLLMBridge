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
        f"Behaviour: {_format_list(profile.get('behaviour_rules', profile.get('response_rules', [])))}\n"
    )

    vocab_text = (
        f"Intents: {_format_list(quest_policy.allowed_intents)}\n"
        f"Topics: {_format_list(quest_policy.allowed_topics)}\n"
        f"Clues: {_format_list(quest_policy.allowed_clues_for_npc(npc_id))}\n"
        f"Response Types: {_format_list(quest_policy.allowed_response_types)}\n"
    )

    npc_policies = quest_policy.policies_for_npc(npc_id)
    policies_text = ""
    for p in npc_policies:
        policies_text += f"- {p['policy_id']}:"
        
        when = p.get("when", {})
        conds = []
        if "intents" in when:
            conds.append(f"intent in [{_format_list(when['intents'])}]")
        if "topics" in when:
            conds.append(f"topic in [{_format_list(when['topics'])}]")
        if "required_clue_not_known" in when:
            conds.append(f"'{when['required_clue_not_known']}' not in known_clues")
        if "required_clue_known" in when:
            conds.append(f"'{when['required_clue_known']}' in known_clues")
            
        if conds:
            policies_text += " if " + " & ".join(conds)
            
        if "response_type" in p:
            policies_text += f" -> requires {p['response_type']}"
            
        policies_text += "\n"

    return f"""Control NPC. Return ONLY JSON.

PROFILE
{profile_text}
VOCAB
{vocab_text}
POLICIES
{policies_text}
SHAPE
{json.dumps(build_output_contract())}

RULES
- Use only VOCAB values.
- Match intent/topic to select ONE policy.
- Keep dialogue short.
- clue_claims must be valid or empty.
- If unclear, use witness_request_clarification.
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
