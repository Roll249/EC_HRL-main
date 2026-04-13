from pathlib import Path
import json

path = Path(r"/home/khang/khang_lab/EC_HRL-main/super_lag.ipynb")
nb = json.loads(path.read_text(encoding='utf-8'))


def cell_by_id(cell_id):
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return cell
    raise ValueError(f"Cell not found: {cell_id}")

helper_source = '''# Toggle this to True to rerun any training block.
RUN_TRAINING = False
from pathlib import Path

WEIGHTS_DIR = Path("weights")
WEIGHT_CHECKPOINTS = {
    "agent_h": (agent_h, "HQ", "HT"),
    "agent_h1": (agent_h1, "HQ1", "HT1"),
    "agent_h2": (agent_h2, "HQ2", "HT2"),
    "agent_m": (agent_m, "MQ", "MT"),
    "agent_m1": (agent_m1, "MQ1", "MT1"),
    "agent_m2": (agent_m2, "MQ2", "MT2"),
    "agent_l": (agent_l, "LQ", "LT"),
    "agent_l1": (agent_l1, "LQ1", "LT1"),
    "agent_l2": (agent_l2, "LQ2", "LT2")
}


def load_pretrained_agents(raise_on_missing=True):
    """Load all checkpoints from WEIGHTS_DIR and return loaded group count."""
    missing_files = []
    for _, (agent, q_name, t_name) in WEIGHT_CHECKPOINTS.items():
        q_path = WEIGHTS_DIR / f"{q_name}.weights.h5"
        t_path = WEIGHTS_DIR / f"{t_name}.weights.h5"
        if not q_path.exists():
            missing_files.append(str(q_path))
        if not t_path.exists():
            missing_files.append(str(t_path))

    if missing_files:
        message = "Missing checkpoint file(s) in 'weights' folder: " + ", ".join(sorted(set(missing_files)))
        if raise_on_missing:
            raise FileNotFoundError(message)
        print(message)
        return 0

    for name, (agent, q_name, t_name) in WEIGHT_CHECKPOINTS.items():
        q_path = WEIGHTS_DIR / f"{q_name}.weights.h5"
        t_path = WEIGHTS_DIR / f"{t_name}.weights.h5"
        agent.load(str(q_path), str(t_path))
        print(f"Loaded {name}: {q_path.name}, {t_path.name}")
    return len(WEIGHT_CHECKPOINTS)
'''

cell_by_id('load-weights-cell')["source"] = [line + "\n" for line in helper_source.splitlines()]

for train_id in [
    '08dbb6e5-478c-4697-9a50-ace06e6f8e93',
    '3419a4b5-9182-4a84-9c40-b9550f61447b',
    '2423f0f5-c6e0-4967-8291-0d6bbec6c264'
]:
    train_cell = cell_by_id(train_id)
    if not train_cell['source'][0].startswith('if RUN_TRAINING:'):
        train_cell['source'] = ['if RUN_TRAINING:\n', '    exec("""\n'] + train_cell['source'] + ['    """\n']

load_cell = cell_by_id('95f1ad62-ac09-4f80-bcd9-04ef70dd5b01')
load_cell['source'] = [
    'loaded_agent_count = load_pretrained_agents(raise_on_missing=True)\n',
    'print(f"Loaded {loaded_agent_count} checkpoint groups from weights folder.")\n'
]

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
