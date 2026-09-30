"""Annotate observed pairs without rewriting actions or synthesizing observations."""
import json
from pathlib import Path

AXES = ("x", "y", "z", "roll", "pitch", "yaw")

class TrajectoryRefiner:
    def annotate(self, records):
        annotated = []
        stats = {"direction_reversals":0, "oscillation_count":0, "redundant_noops":0,
                 "repeated_identical_actions":0, "long_correction_sequences":0}
        previous = None
        previous_reversals = set()
        correction_length = 0
        for row in records:
            r = dict(row)
            a = r.get("executed_semantic_action",r.get("semantic_action"))
            flags = []
            if not r.get("execution_valid") or not a:
                r["annotations"] = flags
                annotated.append(r)
                previous = None
                previous_reversals = set()
                correction_length = 0
                continue
            key = lambda x: tuple(x.get(k) for k in ("mode",*AXES,"gripper"))
            noop = lambda x: all(x.get(k)=="ZERO" for k in AXES) and x.get("gripper")=="HOLD"
            reversals = set()
            if previous:
                reversals = {k for k in AXES if a.get(k) in ("POSITIVE","NEGATIVE") and previous.get(k) in ("POSITIVE","NEGATIVE") and a[k]!=previous[k]}
                for axis in sorted(reversals): flags.append("direction_reversal:"+axis)
                stats["direction_reversals"] += len(reversals)
                for axis in sorted(reversals & previous_reversals): flags.append("oscillation:"+axis)
                stats["oscillation_count"] += len(reversals & previous_reversals)
                if key(a)==key(previous):
                    flags.append("repeated_identical_action")
                    stats["repeated_identical_actions"] += 1
                if noop(a) and noop(previous):
                    flags.append("redundant_noop_candidate")
                    stats["redundant_noops"] += 1
            correction_length = correction_length+1 if reversals else 0
            if correction_length >= 4:
                flags.append("long_correction_sequence")
                stats["long_correction_sequences"] += 1
            r["annotations"] = flags
            annotated.append(r)
            previous = a
            previous_reversals = reversals
        return annotated, stats

    def write(self, directory):
        directory = Path(directory)
        raw = directory / "raw_trajectory.jsonl"
        records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]
        annotated, stats = self.annotate(records)
        (directory/"annotated_trajectory.jsonl").write_text("".join(json.dumps(r)+"\n" for r in annotated),encoding="utf-8")
        (directory/"refinement_stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
        return stats
