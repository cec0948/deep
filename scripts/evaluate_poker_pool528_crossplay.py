"""Directed one-versus-five cross-play of three seed-matched checkpoint policies."""
import argparse
from contextlib import closing
from copy import copy
import gzip
import json
import math
import os
from pathlib import Path
import random
from statistics import NormalDist
import time

import numpy as np

from audit_postflop_averaging import terminal_auditor, verify_terminals
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import (
    _validate_reply, hand_seed, policy_dependency_hashes, semantic_seed,
)
from evaluate_poker_postflop_snapshots import BASE, MIX, load_candidate
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import evaluator_policy
from poker_reference_opponents import checked_action
from run_poker_preflop_confirmation import run_jobs


STUDY_PATH = Path("preflop-followup/480-v1/pool-extension-528-20261126")
STUDY = ROOT / "artifacts" / STUDY_PATH
OUTPUT = STUDY / "crossplay"
PUBLIC = ROOT / "reports" / STUDY_PATH / "crossplay"
ORIGINAL = STUDY.parent / "opponent-diversity-20261123"
PREVIOUS_PLAN = ORIGINAL / "population/evaluation-plan.json"
DESIGN_SHA = "5fb97eaae9ecafef9a8a1a0d7b4b2053664f86aaaabca2f0334ee8ea36af4288"
PREVIOUS_SHA = "a851844aee89e4032992840e40b2bf41ac765cc132dd5be9f94fee09e9540b43"
PARENT_STAGE_SHA = "8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9"
LABELS = ("specialist480", "mixture504", "mixture528")
SNAPSHOTS = {"mixture504": (486, 492, 498, 504), "mixture528": (510, 516, 522, 528)}
SEEDS = (342, 343, 344)
START, DEALS = 2060300000, 1000
SCRIPT = Path(__file__).resolve()


def matrix_jobs():
    return [(h, v, s) for h in LABELS for v in LABELS for s in SEEDS]


def cell_id(hero, others, seed):
    return f"{hero}-vs-{others}-seed-{seed}"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(ref, label, source):
    return load_candidate(BASE if label == LABELS[0] else MIX, ref, source)


def bind():
    verify_files({OUTPUT / "design.json": DESIGN_SHA, PUBLIC / "design.json": DESIGN_SHA,
                  PREVIOUS_PLAN: PREVIOUS_SHA, ORIGINAL / "stage-ready.json": PARENT_STAGE_SHA})
    design = read(OUTPUT / "design.json")
    require((design["card_start"], design["deals"], design["seeds"], design["stages"]) ==
            (START, DEALS, list(SEEDS), list(LABELS)), "Cross-play differs from reserved design")
    previous, stage = read(PREVIOUS_PLAN), read(STUDY / "stage-ready.json")
    require(stage["training_complete"] and stage["iteration"] == 528, "Missing final checkpoint")
    refs = {LABELS[0]: previous["policies"]["specialist480"],
            LABELS[1]: previous["policies"]["diverse504"], LABELS[2]: stage["policies"]["mixture528"]}
    root_config = ROOT / "configs/poker-pool-extension-528-20261126.json"
    registration = read(STUDY / "registration.json")
    require(registration["evaluation_cards"]["crossplay"] == [START, DEALS]
            and registration["final_iteration"] == 528, "Master study registration changed")
    verify_files({root_config:registration["config_sha256"]})
    files = [SCRIPT, root_config, STUDY / "registration.json", OUTPUT / "design.json", PREVIOUS_PLAN, STUDY / "stage-ready.json",
             ORIGINAL / "stage-ready.json", terminal_auditor(), ROOT / "native/noregrets-json/terminal_audit.rs"]
    files += [ROOT / "scripts" / name for name in ("evaluate_poker_postflop_snapshots.py",
              "poker_reference_opponents.py", "audit_postflop_averaging.py", "run_poker_preflop_confirmation.py")]
    dependencies = {str(p): digest(p) for p in files}
    for s in SEEDS:
        parent = read(refs[LABELS[0]][str(s)]["path"])
        for label in LABELS:
            ref = refs[label][str(s)]
            policy = load(ref, label, previous["source"])
            if label in SNAPSHOTS:
                spec = read(ref["path"])
                require(spec["seed"] == s and tuple(row["iteration"] for row in spec["postflop"]) == SNAPSHOTS[label]
                        and spec["preflop"]["sha256"] == parent["preflop"]["sha256"], "Wrong mixture binding")
            dependencies.update(policy_dependency_hashes(policy, previous["source"]))
    verify_files(dependencies)
    with (OUTPUT / "binding.lock").open("x"):
        plan_path = OUTPUT / "evaluation-plan.json"
        atomic_json(plan_path, dict(registered_utc=utc(), design_sha256=DESIGN_SHA, source=previous["source"],
                                    policies=refs, frozen_files=dependencies, jobs=matrix_jobs()))
        receipt = dict(bound_utc=utc(), plan_sha256=digest(plan_path), design_sha256=DESIGN_SHA,
                       stage_sha256=digest(STUDY / "stage-ready.json"))
        atomic_json(OUTPUT / "bound.json", receipt)
        atomic_json(PUBLIC / "bound.json", receipt)


def plan():
    path = OUTPUT / "evaluation-plan.json"
    key = read(OUTPUT / "bound.json")["plan_sha256"]
    verify_files({path: key, OUTPUT / "design.json": DESIGN_SHA})
    return read(path), key


def private_slot(card, seat):
    key = hand_seed(card, seat).to_bytes(8, "big")
    return random.Random(b"scheduled-postflop-choice-v1\0" + key).randrange(4)


def play(engine, policies, labels, card, hero):
    state = engine.call(dict(op="reset", seed=card, button=0))
    slots = []
    for seat, (p, label) in enumerate(zip(policies, labels, strict=True)):
        if label in SNAPSHOTS:
            p.begin_hand(hand_seed(card, seat))
            slot = p.postflop.index(p.active)
            require(slot == private_slot(card, seat), "Per-seat private slot mismatch")
            slots.append(slot)
        else:
            slots.append(None)
    actions = []
    while not state["terminal"]:
        actor = state["seat_to_act"]
        seed = semantic_seed(card, actor, state)
        answer = _validate_reply(state, policies[actor].decide(state, seed))
        action = dict(checked_action(state, answer["action"]), seat=actor)
        actions.append(action)
        require(len(actions) <= 512, "Excessive action count")
        state = engine.call(dict(op="step", action=action))
    utilities = state["utilities"]
    require(len(utilities) == 6 and all(type(x) is int for x in utilities) and sum(utilities) == 0,
            "Invalid terminal chip ledger")
    return dict(seed=card, deal=card-START, hero=hero, hand_seed=hand_seed(card, hero),
                chips=utilities[hero], utilities=utilities, slots=slots, steps=len(actions), actions=actions)


def seated(roles, hero):
    # Private mixture state (.active) lives on a distinct runtime object for
    # every seat. Only read-only action networks are shared across these copies.
    return [roles[seat == hero][seat] for seat in range(6)]


def replay_networks(content, label, seed):
    """Load bare neural policies, independently of the evaluated mixture/selector."""
    path = Path(content["policies"][label][str(seed)]["path"])
    spec = read(path)
    if label in SNAPSHOTS:
        parts = [spec["preflop"], *spec["postflop"]]
        paths = [path.parent / item["file"] for item in parts]
    else:
        parts = [spec["preflop"], spec["postflop"]]
        paths = [Path(item["path"]) for item in parts]
    verify_files({p:item["sha256"] for p,item in zip(paths,parts,strict=True)})
    return [evaluator_policy(p,content["source"]) for p in paths]


def replay_choices(row, labels, networks, engine):
    card = row["seed"]
    state = engine.call(dict(op="reset", seed=card, button=0))
    for logged in row["actions"]:
        require(not state["terminal"], "Raw action after settlement")
        actor = state["seat_to_act"]
        label = labels[actor]
        pick = 0
        if state["street"] != "preflop":
            pick = 1 + (private_slot(card,actor) if label in SNAPSHOTS else 0)
        policy = networks[label][pick]
        reply = _validate_reply(state, policy.decide(state,semantic_seed(card,actor,state)))
        action = dict(checked_action(state,reply["action"]),seat=actor)
        require(action == logged, "Independently selected snapshot or action differs")
        state = engine.call(dict(op="step",action=logged))
    require(state["terminal"] and state["utilities"] == row["utilities"], "Independent payoff differs")


def worker(hero, others, training_seed):
    content, key = plan()
    require((hero,others,training_seed) in matrix_jobs(), "Unexpected matrix assignment")
    name = cell_id(hero, others, training_seed)
    folder = OUTPUT / "evaluation" / name
    folder.mkdir()
    def status(phase, **kw):
        atomic_json(folder / "status.json", dict(phase=phase, utc=utc(), pid=os.getpid(),
                    process_start=process_start(os.getpid()), **kw))
    try:
        status("loading")
        dependencies = content["frozen_files"]
        verify_files(dependencies)
        policies = {label: load(content["policies"][label][str(training_seed)], label, content["source"])
                    for label in set((hero, others))}
        roles = [[copy(policies[label]) for _ in range(6)] for label in (others, hero)]
        role_labels = [[others]*6, [hero]*6]
        card_blocks = []
        n_actions = 0
        raw = folder / "hands.jsonl.gz"
        deadline = time.monotonic() + 7200
        with closing(Bridge()) as engine, gzip.open(raw, "xt", compresslevel=1) as dest:
            for card in range(START, START + DEALS):
                require(time.monotonic() < deadline, "Cross-play cell exceeded its time budget")
                scores = []
                first = None
                for seat in range(6):
                    row = play(engine, seated(roles, seat), seated(role_labels, seat), card, seat)
                    if hero == others:
                        transcript = (row["actions"], row["utilities"], row["slots"])
                        if first is None:
                            first = transcript
                        require(first == transcript, "Self-play changed when only the scoring seat changed")
                    dest.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
                    scores.append(row["chips"])
                    n_actions += row["steps"]
                card_blocks.append(sum(scores)/6)
                if (card-START+1) % 100 == 0:
                    status("evaluating", completed_card_blocks=card-START+1)
        status("replaying")
        proof = verify_terminals(raw)
        require(proof == {"hands":6*DEALS,"decisions":n_actions}, "Incomplete replay")
        verify_files(dependencies)
        result = dict(complete=True, completed_utc=utc(), design_sha256=DESIGN_SHA, plan_sha256=key,
                      hero=hero, opponents=others, seed=training_seed, start=START, deals=DEALS,
                      hands=6*DEALS, terminal_replay=proof, cards_bb100=card_blocks,
                      raw_sha256=digest(raw), hero_policy_sha256=content["policies"][hero][str(training_seed)]["sha256"],
                      opponent_policy_sha256=content["policies"][others][str(training_seed)]["sha256"])
        atomic_json(folder / "results.json", result)
        status("complete", result_sha256=digest(folder/"results.json"))
    except BaseException as exc:
        status("failed", reason=f"{type(exc).__name__}: {exc}", automatic_retry=False)
        raise


def audit_cell(content, key, h, v, seed, engine):
    name = cell_id(h, v, seed)
    folder = OUTPUT / "evaluation" / name
    raw = folder / "hands.jsonl.gz"
    result = read(folder/"results.json")
    require((result["complete"], result["design_sha256"], result["plan_sha256"], result["hero"],
             result["opponents"], result["seed"], result["start"], result["deals"], result["hands"],
             result["hero_policy_sha256"], result["opponent_policy_sha256"]) ==
            (True, DESIGN_SHA, key, h, v, seed, START, DEALS, DEALS*6, content["policies"][h][str(seed)]["sha256"],
             content["policies"][v][str(seed)]["sha256"]), "Cross-play cell belongs to another comparison")
    verify_files({raw:result["raw_sha256"]})
    networks = {label: replay_networks(content,label,seed) for label in set((h,v))}
    role_labels = [[v]*6,[h]*6]
    sums = np.zeros(DEALS, dtype=np.int64)
    rows = steps = reproduced = 0
    with gzip.open(raw, "rt") as tape:
        for line in tape:
            row = json.loads(line)
            index, seat = divmod(rows,6)
            card = START+index
            labels = seated(role_labels, seat)
            require(index < DEALS and (row["deal"],row["seed"],row["hero"],row["hand_seed"]) ==
                    (index,card,seat,hand_seed(card,seat)), "Raw deal/seat pairing failed")
            slots = [private_slot(card,s) if label in SNAPSHOTS else None for s,label in enumerate(labels)]
            require(row["slots"] == slots and row["steps"] == len(row["actions"]) and
                    row["chips"] == row["utilities"][seat] and sum(row["utilities"]) == 0,
                    "Raw private tapes or chip ledger differ")
            if index % 199 == 0 or index+1 == DEALS:
                replay_choices(row,labels,networks,engine)
                reproduced += 1
            sums[index] += row["chips"]
            rows += 1
            steps += row["steps"]
    blocks = sums/6.
    require(verify_terminals(raw) == {"hands":rows,"decisions":steps} == result["terminal_replay"]
            and rows == DEALS*6 and np.array_equal(blocks, result["cards_bb100"]),
            "Independent terminal/card summary did not reproduce")
    require(h != v or not np.any(blocks), "Self-play was not zero on each paired card")
    return blocks, dict(result_sha256=digest(folder/"results.json"), raw_sha256=result["raw_sha256"],
                        terminal_hands=rows, terminal_decisions=steps, policy_resampled_hands=reproduced)


def estimate(values, tests):
    # rows are fixed training seeds, columns are the independent deals.
    require(values.shape==(len(SEEDS),DEALS) and np.isfinite(values).all(), "Wrong card matrix")
    cards = values.mean(axis=0)
    mean = float(cards.mean())
    se = cards.std(ddof=1)/math.sqrt(len(cards))
    width = float(NormalDist().inv_cdf(1-.05/(2*tests))*se)
    return dict(mean=mean, lower=mean-width, upper=mean+width,
                by_seed={str(s):float(row.mean()) for s,row in zip(SEEDS,values,strict=True)})


def summarize():
    content, key = plan()
    verify_files(content["frozen_files"])
    cells, proofs = {}, {}
    with closing(Bridge()) as engine:
        for h, v, s in matrix_jobs():
            row, proof = audit_cell(content,key,h,v,s,engine)
            cells.setdefault((h,v),[]).append(row)
            proofs[cell_id(h,v,s)] = proof
    cells = {pair:np.asarray(arrays) for pair, arrays in cells.items()}
    matrix = {h:{v:estimate(cells[h,v],6) for v in LABELS} for h in LABELS}
    diffs = {v:estimate(cells[LABELS[2],v]-cells[LABELS[1],v],3) for v in LABELS}
    reciprocal = {f"{a}_vs_{b}":estimate(cells[a,b]-cells[b,a],3)
                  for i,a in enumerate(LABELS) for b in LABELS[i+1:]}
    dominance = {(a,b) for a in LABELS for b in LABELS if a!=b and
                 matrix[a][b]["lower"] > 0 and matrix[b][a]["upper"] < 0}
    cycles = []
    a,b,c = LABELS
    for order in ((a,b,c),(a,c,b)):
        if all((x,y) in dominance for x,y in zip(order,(*order[1:],order[0]),strict=True)):
            cycles.append(list(order))
    report = dict(complete=True, verified_utc=utc(), design_sha256=DESIGN_SHA, plan_sha256=key,
                  completed_cells=len(proofs), terminal_replay_hands=sum(x["terminal_hands"] for x in proofs.values()),
                  terminal_replay_decisions=sum(x["terminal_decisions"] for x in proofs.values()),
                  policy_resampled_hands=sum(x["policy_resampled_hands"] for x in proofs.values()),
                  units="bb/100", fixed_seeds=list(SEEDS), original_card_blocks=DEALS,
                  matrix=matrix, new528_minus_parent504_vs_fixed_opponent=diffs,
                  directed_payoff_minus_reverse=reciprocal,
                  supported_dominance_edges=sorted(dominance), supported_cycles=cycles,
                  intervals="Matched original-card normal intervals. Bonferroni 95% within each registered family.",
                  scope="One seat against five identically specified but independently acting seats. "
                        "The reverse lineup is a different six-player game, not an antisymmetric heads-up payoff. "
                        "Self cells are identically zero from paired tapes. Missing significant edges cannot rule out cycling.",
                  six_player_exploitability=None, best_response_computed=False,
                  result_sha256={c:p["result_sha256"] for c,p in proofs.items()})
    require(report["terminal_replay_hands"] == DEALS*6*len(matrix_jobs()),"Incomplete matrix")
    verify_files(content["frozen_files"])
    atomic_json(OUTPUT / "audit.json",proofs)
    for directory in (OUTPUT,PUBLIC):
        atomic_json(directory / "summary.json",report)
    return report


def run(workers):
    plan()
    marker = OUTPUT/"launch.json"
    with marker.open("x") as handle:
        json.dump(dict(utc=utc(),pid=os.getpid(),process_start=process_start(os.getpid())),handle)
    (OUTPUT/"evaluation").mkdir()
    jobs = {cell_id(h,v,s): [str(PYTHON),str(SCRIPT),"worker",h,v,str(s)] for h,v,s in matrix_jobs()}
    run_jobs(OUTPUT,jobs,workers,"evaluation")
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode",required=True)
    modes.add_parser("bind")
    p = modes.add_parser("run")
    p.add_argument("--workers",type=int,choices=(1,2,3,4),required=True)
    p = modes.add_parser("worker")
    p.add_argument("hero",choices=LABELS)
    p.add_argument("opponents",choices=LABELS)
    p.add_argument("seed",type=int,choices=SEEDS)
    modes.add_parser("summarize")
    args = parser.parse_args()
    if args.mode == "bind":
        bind()
    elif args.mode == "run":
        run(args.workers)
    elif args.mode == "worker":
        worker(args.hero,args.opponents,args.seed)
    else:
        summarize()
