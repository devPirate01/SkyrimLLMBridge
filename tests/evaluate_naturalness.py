import csv, json, os, pathlib
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BRIDGE_DIR / 'evaluation_results'

def compile_dialogues():
    jsonl_files = sorted(EVAL_DIR.glob('benchmark_*.jsonl'))
    records = []
    for fpath in jsonl_files:
        run_name = fpath.stem.replace('benchmark_', '')
        lines = [json.loads(x) for x in fpath.read_text(encoding='utf-8').splitlines() if x.strip()]
        for rec in lines:
            raw_out = rec.get('raw_output', '')
            dialogue = ''
            if raw_out:
                try:
                    parsed = json.loads(raw_out)
                    dialogue = parsed.get('dialogue', '')
                except:
                    dialogue = raw_out
            records.append({
                'model_run': run_name,
                'test_case': rec.get('test_case', ''),
                'player_message': rec.get('player_message', ''),
                'known_clues': str(rec.get('known_clues', [])),
                'expected_policy': rec.get('expected_policy', ''),
                'actual_policy': rec.get('actual_policy', ''),
                'dialogue': dialogue,
                'strict_json_valid': rec.get('strict_json_valid', False),
            })
    
    out_csv = EVAL_DIR / 'dialogue_comparison_matrix.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model_run', 'test_case', 'player_message', 'known_clues',
            'expected_policy', 'actual_policy', 'dialogue', 'strict_json_valid'
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f'Successfully compiled {len(records)} dialogue evaluations to: {out_csv}')

if __name__ == '__main__':
    compile_dialogues()
