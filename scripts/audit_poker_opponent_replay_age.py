"""Measure exposure to the 481–504 collection window in the final fitted replay."""
import json
from pathlib import Path

import numpy as np

from fullhouse_v3_common import ROOT, atomic_json, digest
from regret_partition import base_strata


STUDIES = ROOT / "artifacts/preflop-followup/480-v1"
SOURCE = STUDIES / "opponent-diversity-20261123"
STAGE_HASH = "8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9"


def read(path):
    return json.loads(path.read_bytes())


def summarize_groups(groups, iterations, selected):
    group_ids = np.unique(groups[selected])
    recent = iterations > 480
    # ConditionalRegretSampler gives groups equal probability; within each
    # group, probability is proportional to the collection iteration.
    weighted = []
    for group in group_ids:
        weights = iterations[groups == group]
        weighted.append(float(weights[weights > 480].sum() / weights.sum()))
    count = int(np.count_nonzero(selected))
    recent_count = int(np.count_nonzero(recent & selected))
    return dict(rows=count, recent_rows=recent_count, recent_fraction=recent_count / count,
                recent_draw_probability=float(np.mean(weighted)))


def seed_report(seed):
    run = SOURCE / "pool504" / f"seed-{seed}"
    saved = read(run / "current.json")
    if saved["iteration"] != 504:
        raise ValueError("Expected final 504 checkpoint")
    checkpoint = run / "checkpoints" / saved["directory"]
    for name in ("state.json", "arrays.npz"):
        if digest(checkpoint / name) != saved["files"][name]:
            raise ValueError(f"Checkpoint data differs for {seed}/{name}")
    state = dict(read(checkpoint / "state.json")["items"])
    groups, iterations = [], []
    with np.load(checkpoint / "arrays.npz", allow_pickle=False) as arrays:
        for item in state["buffers"]["items"]:
            buf = dict(item["items"])
            count = buf["count"]["value"]
            groups.append(base_strata(arrays[buf["features"]["key"]][:count]))
            iterations.append(arrays[buf["iterations"]["key"]][:count].astype(np.float64))
    group = np.concatenate(groups)
    iteration = np.concatenate(iterations)
    if not np.all((iteration >= 1) & (iteration <= 504) & (iteration == np.floor(iteration))):
        raise ValueError("Invalid collection iteration")
    final_round = json.loads((run / "diagnostics.jsonl").read_bytes().splitlines()[-1])
    streets = [summarize_groups(group, iteration, (group >= start) & (group < start + 4))
               for start in (10, 14, 18)]
    if final_round["iteration"] != 504 or [s["rows"] for s in streets] != final_round["fit_rows_by_street"]:
        raise ValueError("Street counts disagree with the final fit")
    if not np.array_equal(np.unique(group[group >= 10]), np.arange(10, 22)):
        raise ValueError("Unexpected active postflop groups")
    return dict(postflop=summarize_groups(group, iteration, group >= 10),
                **dict(zip(("flop", "turn", "river"), streets, strict=True)))


if __name__ == "__main__":
    if digest(SOURCE / "stage-ready.json") != STAGE_HASH:
        raise ValueError("Wrong treatment stage")
    result = dict(source_stage_sha256=STAGE_HASH, checkpoint_iteration=504,
                  new_collection_iterations=[481, 504],
                  scope="Final fitting distribution; actual draw counts by collection iteration were not logged.",
                  by_seed={str(s): seed_report(s) for s in (342, 343, 344)})
    path = Path("preflop-followup/480-v1/flop-rollback-20261123/replay-age.json")
    for folder in ("artifacts", "reports"):
        atomic_json(ROOT / folder / path, result)
