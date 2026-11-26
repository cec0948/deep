"""Bound a deviation gain for three tested strategies against each symmetric table."""
from evaluate_poker_pool528_crossplay import DESIGN_SHA, LABELS, OUTPUT, PUBLIC, read, require
from fullhouse_v3_common import atomic_json, digest


def score(summary):
    require(summary["complete"] and summary["design_sha256"] == DESIGN_SHA
            and summary["completed_cells"] == 27, "Incomplete registered cross-play")
    result = {}
    for opponent in LABELS:
        column = {hero:summary["matrix"][hero][opponent] for hero in LABELS}
        require(all(column[opponent][key] == 0 for key in ("lower", "mean", "upper")),
                "A symmetric table did not give the seat-averaged zero payoff")
        # Monotonicity extends the original simultaneous cell bounds to all maxima.
        result[opponent] = {key:max(row[key] for row in column.values()) for key in ("mean","lower","upper")}
        result[opponent]["highest_by_estimate"] = max(column, key=lambda hero:column[hero]["mean"])
    return result


if __name__ == "__main__":
    path = OUTPUT/"summary.json"
    report = dict(matrix_sha256=digest(path), units="bb/100",
                  restricted_one_seat_deviation_gain=score(read(path)),
                  scope="Against five copies of a fixed checkpoint, one seat may choose any one of the "
                        "three tested checkpoint policies. Average over seats and fixed training seeds "
                        "before choosing the checkpoint. All-self gives zero. Bounds propagate the "
                        "simultaneous 95% matrix intervals and cover only the finite set of deviations.",
                  unrestricted_best_response_computed=False, six_player_exploitability=None)
    for folder in (OUTPUT, PUBLIC):
        atomic_json(folder/"restricted-deviations.json",report)
