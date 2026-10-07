"""
tests/run_frozen_evaluation.py

Purpose:
    Final dissertation evaluation runner.
    Loads the frozen 100-case test set and runs each model configuration
    across 5 predefined seeds, producing 500 outputs per model (2,000 total).

Usage:
    python tests/run_frozen_evaluation.py
    (The script will ask you to type a label for the current model)

Output:
    evaluation_results/frozen_eval_<MODEL_LABEL>_<timestamp>.jsonl

Note on TTFT:
    The current API client uses synchronous (non-streaming) requests.
    total_latency_s therefore approximates end-to-end latency.
    For the dissertation, this is documented as End-to-End Latency.
"""

import json
import os
import sys
import datetime
import statistics
from pathlib import Path

from dotenv import load_dotenv

BRIDGE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BRIDGE_DIR))

from llm_payload_builder import build_structured_messages, build_turn_state
from llm_reply_processor import StructuredResponseError, validate_structured_response
from data_director import QuestPolicy, load_profile
from llm_api_client import call_llm, load_model_parameters, provider_settings, LLMAPIError

load_dotenv(BRIDGE_DIR / ".env", override=True)

LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "unsloth").strip().lower()
KOBOLDCPP_URL   = os.getenv("KOBOLDCPP_URL",   "http://127.0.0.1:5001/v1").rstrip("/")
KOBOLDCPP_MODEL = os.getenv("KOBOLDCPP_MODEL", "koboldcpp").strip()
OPENWEBUI_URL   = os.getenv("OPENWEBUI_URL",   "").rstrip("/")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN", "").strip()
OPENWEBUI_MODEL = os.getenv("OPENWEBUI_MODEL", "llama3.3:latest").strip()

SEEDS           = [42, 101, 202, 303, 404]
TIMEOUT_SECONDS = 180
QUEST_STAGE     = 10
NPC_ID          = "tavern_witness"
LOCATION        = "The Bannered Mare"

REQUIRED_KEYS = [
    "schema_version", "dialogue", "player_intent", "topic",
    "policy_id", "response_type", "clue_claims", "action_request",
]


def check_strict_raw_json(raw_output):
    if not raw_output or not raw_output.strip():
        return False, False, "empty_response", None
    stripped = raw_output.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        if "```" in stripped:
            return False, False, "markdown_fence", None
        return False, False, "prefix_or_suffix_text", None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        return False, False, f"json_parse_failure: {e}", None
    if not isinstance(data, dict):
        return False, False, "not_a_json_object", None
    keys = list(data.keys())
    if set(keys) != set(REQUIRED_KEYS):
        missing = [k for k in REQUIRED_KEYS if k not in keys]
        extra   = [k for k in keys if k not in REQUIRED_KEYS]
        if missing:
            return False, False, f"missing_key: {missing}", None
        if extra:
            return False, False, f"extra_key: {extra}", None
    if keys != REQUIRED_KEYS:
        return False, True, "wrong_key_order", data
    if data.get("schema_version") != "1.0":
        return False, False, "wrong_schema_version", data
    if data.get("action_request") is not None:
        return False, False, "wrong_null_type", data
    if not isinstance(data.get("clue_claims"), list):
        return False, False, "wrong_clue_claims_type", data
    return True, True, None, data


def run_case(case, profile, policy, url, model, headers, model_parameters, seed, run_id, model_label):
    recent_messages = []
    if case.get("multi_turn") and case.get("setup_player_message"):
        setup_state = build_turn_state(
            request_id=f"{case['id']}_setup",
            player_input=case["setup_player_message"],
            location=LOCATION,
            quest_stage=QUEST_STAGE,
            known_clues=case["known_clues"],
            session_id=run_id,
            experimental_group=model_label,
            npc_id=NPC_ID,
        )
        recent_messages = [
            {"role": "user", "content": json.dumps(setup_state, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({
                "schema_version": "1.0",
                "dialogue": "[Setup turn]",
                "player_intent": "offer_information",
                "topic": "witness",
                "policy_id": case["setup_expected_policy"],
                "response_type": "acknowledge_information",
                "clue_claims": [],
                "action_request": None,
            }, ensure_ascii=False)},
        ]

    turn_state = build_turn_state(
        request_id=case["id"],
        player_input=case["message"],
        location=LOCATION,
        quest_stage=QUEST_STAGE,
        known_clues=case["known_clues"],
        session_id=run_id,
        experimental_group=model_label,
        npc_id=NPC_ID,
    )
    messages = build_structured_messages(
        profile=profile,
        quest_policy=policy,
        turn_state=turn_state,
        npc_id=NPC_ID,
        recent_messages=recent_messages or None,
    )

    seeded_params = dict(model_parameters)
    seeded_params["seed"] = seed

    record = {
        "run_id": run_id, "model_label": model_label, "seed": seed,
        "case_id": case["id"], "category": case.get("category", "Unknown"),
        "multi_turn": case.get("multi_turn", False),
        "player_message": case["message"], "known_clues": case["known_clues"],
        "expected_policy": case["expected_policy"],
        "strict_json_valid": False, "schema_valid": False,
        "raw_policy_correct": False, "raw_clue_leak_detected": False,
        "validator_accepted": False, "validated_policy_correct": False,
        "overall_correct": False,
        "total_latency_s": 0.0,
        "error_type": None, "error_message": None, "raw_output": "",
    }

    try:
        raw_output, latency = call_llm(
            provider=LLM_PROVIDER, url=url, model="default", headers=headers,
            messages=messages, parameters=seeded_params, timeout_seconds=TIMEOUT_SECONDS,
        )
        record["total_latency_s"] = round(latency, 4)
        record["raw_output"] = raw_output

        strict_valid, schema_valid, fmt_error, parsed = check_strict_raw_json(raw_output)
        record["strict_json_valid"] = strict_valid
        record["schema_valid"] = schema_valid
        if fmt_error:
            record["error_type"] = fmt_error

        if parsed:
            actual_policy = parsed.get("policy_id", "")
            actual_clues = parsed.get("clue_claims", [])
            record["raw_policy_correct"] = (actual_policy == case["expected_policy"])
            clue_permitting_policies = ["witness_reveal_c1", "witness_repeat_c1"]
            record["raw_clue_leak_detected"] = (
                "C1" in actual_clues and actual_policy not in clue_permitting_policies
            )

        try:
            validated = validate_structured_response(
                raw_text=raw_output, quest_policy=policy, npc_id=NPC_ID,
                quest_stage=QUEST_STAGE, known_clues=case["known_clues"],
            )
            record["validator_accepted"] = True
            record["validated_policy_correct"] = (validated.policy_id == case["expected_policy"])
            record["overall_correct"] = strict_valid and record["validated_policy_correct"]
        except StructuredResponseError as ve:
            record["validator_accepted"] = False
            record["error_type"] = record["error_type"] or "validation_error"
            record["error_message"] = str(ve)

    except LLMAPIError as api_err:
        record["error_type"] = "api_failure"
        record["error_message"] = str(api_err)
    except Exception as err:
        record["error_type"] = "critical_error"
        record["error_message"] = str(err)

    return record


def main():
    frozen_path = BRIDGE_DIR / "tests" / "frozen_test_set_100.json"
    if not frozen_path.exists():
        print(f"ERROR: {frozen_path} not found. Run build_frozen_test_set.py first.")
        sys.exit(1)

    with open(frozen_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    print("=" * 72)
    print("FROZEN EVALUATION — DISSERTATION FINAL BENCHMARK")
    print("=" * 72)
    
    # INTERACTIVE PROMPT FOR MODEL LABEL
    print("What model is currently loaded in Unsloth?")
    print("Examples: Base_E2B, Finetuned_E2B, Base_26B_MoE")
    model_label = input("Type label here: ").strip().replace(" ", "_")
    if not model_label:
        model_label = "Unknown_Model"

    profile = load_profile("tavern_witness")
    policy  = QuestPolicy()
    url, _, headers = provider_settings(
        LLM_PROVIDER, KOBOLDCPP_URL, KOBOLDCPP_MODEL,
        OPENWEBUI_URL, OPENWEBUI_TOKEN, OPENWEBUI_MODEL,
    )
    # Load parameters, falling back to default if specific model json isn't found
    model_parameters = load_model_parameters("default")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id    = f"frozen_{model_label}_{timestamp}"
    eval_dir  = BRIDGE_DIR / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path  = eval_dir / f"frozen_eval_{model_label}_{timestamp}.jsonl"

    total_runs = len(test_cases) * len(SEEDS)

    print("\n" + "=" * 72)
    print(f"Run ID:        {run_id}")
    print(f"Model Label:   {model_label}")
    print(f"Provider:      {LLM_PROVIDER} -> {url}")
    print(f"Cases:         {len(test_cases)}  x  Seeds: {SEEDS}  =  {total_runs} outputs")
    print(f"Output file:   {out_path.name}")
    print("=" * 72)
    print("CHECKLIST before proceeding:")
    print("  [ ] Skyrim is open and Runa is visible in The Bannered Mare")
    print("  [ ] FPS overlay is active and logging")
    print("  [ ] Unsloth is running with the correct model loaded")
    input("\nPress ENTER to start the evaluation...\n")

    all_records = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for seed_idx, seed in enumerate(SEEDS, 1):
            print(f"\n{'='*72}")
            print(f"  SEED {seed}  (run {seed_idx} of {len(SEEDS)})")
            print(f"{'='*72}")
            for idx, case in enumerate(test_cases, 1):
                rec = run_case(
                    case, profile, policy, url, "default", headers,
                    model_parameters, seed, run_id, model_label,
                )
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                status = "PASS" if rec["overall_correct"] else "FAIL"
                leak   = " !!LEAK!!" if rec["raw_clue_leak_detected"] else ""
                print(
                    f"  [{seed}] {idx:03d}/{len(test_cases)}  {case['id']:<12}  "
                    f"{status}  raw={'Y' if rec['raw_policy_correct'] else 'N'}  "
                    f"val={'Y' if rec['validator_accepted'] else 'N'}  "
                    f"{rec['total_latency_s']:.2f}s{leak}"
                )

    n = len(all_records)
    latencies = [r["total_latency_s"] for r in all_records if r["total_latency_s"] > 0]

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"Total outputs:                    {n}")
    print(f"Strict JSON valid (pre-val):      {sum(r['strict_json_valid'] for r in all_records)}/{n}  ({sum(r['strict_json_valid'] for r in all_records)/n*100:.1f}%)")
    print(f"Policy correct (pre-val):         {sum(r['raw_policy_correct'] for r in all_records)}/{n}  ({sum(r['raw_policy_correct'] for r in all_records)/n*100:.1f}%)")
    print(f"Validator accepted (post-val):    {sum(r['validator_accepted'] for r in all_records)}/{n}  ({sum(r['validator_accepted'] for r in all_records)/n*100:.1f}%)")
    print(f"Policy correct (post-val):        {sum(r['validated_policy_correct'] for r in all_records)}/{n}  ({sum(r['validated_policy_correct'] for r in all_records)/n*100:.1f}%)")
    print(f"Overall correct:                  {sum(r['overall_correct'] for r in all_records)}/{n}  ({sum(r['overall_correct'] for r in all_records)/n*100:.1f}%)")
    print(f"Clue leakage events:              {sum(r['raw_clue_leak_detected'] for r in all_records)}/{n}")
    print(f"API failures:                     {sum(r['error_type']=='api_failure' for r in all_records)}")
    if latencies:
        print(f"Latency mean/median/P95:          {statistics.mean(latencies):.2f}s / {statistics.median(latencies):.2f}s / {(statistics.quantiles(latencies,n=20)[18] if len(latencies)>=20 else max(latencies)):.2f}s")
    print("=" * 72)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
