"""Aggregate actual rollout episodes only; exclude diagnostics and calibration."""
import csv
import json
from collections import Counter
from pathlib import Path
from src.trajectory.refiner import TrajectoryRefiner


def analyze(root="outputs",output="results.csv"):
    rows=[]
    for bucket in ("success","failure"):
        for p in sorted((Path(root)/bucket).glob("**/metadata.json")):
            m=json.loads(p.read_text(encoding="utf-8"))
            if m.get("category")!="rollout" or m.get("dry_run"): continue
            records=[json.loads(x) for x in (p.parent/"trajectory.jsonl").read_text(encoding="utf-8").splitlines() if x]
            executed=[r for r in records if r.get("execution_valid")]
            modes=Counter(r["semantic_action"]["mode"] for r in executed)
            grippers=Counter(r.get("executed_semantic_action",r["semantic_action"])["gripper"] for r in executed)
            actions=Counter(json.dumps({k:v for k,v in r["semantic_action"].items() if k!="reason"},sort_keys=True) for r in executed)
            latencies=[a["latency"] for r in records for a in r.get("attempts",[])]
            _,stats=TrajectoryRefiner().annotate(records)
            row={"episode":str(p.parent),"task_id":m["task_id"],"seed":m["environment_seed"],
                "success":int(m["success"]),"environment_steps":m["executed_steps"],"semantic_decisions":m.get("semantic_decisions",m["executed_steps"]),"astra_calls":m["astra_calls"],
                "rollout_seconds":m["rollout_duration"],"mean_astra_latency":sum(latencies)/len(latencies) if latencies else 0,
                "invalid_output_rate":m["invalid_outputs"]/m["astra_calls"] if m["astra_calls"] else 0,
                "semantic_action_distribution":json.dumps(actions),"termination":m["termination"],"total_grasp_attempts":m.get("total_grasp_attempts",0),"grasp_recovery_count":m.get("grasp_recovery_count",0),**stats}
            for key in ("COARSE","NORMAL","FINE"):
                row[key+"_pct"]=100*modes[key]/len(executed) if executed else 0
            for key in ("OPEN","HOLD","CLOSE"):
                row["gripper_"+key+"_count"]=grippers[key]
                row["gripper_"+key+"_pct"]=100*grippers[key]/len(executed) if executed else 0
            rows.append(row)
    target=Path(output); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]) if rows else ["episode","success","environment_steps","astra_calls"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary={"episodes":len(rows),"success_rate":sum(r["success"] for r in rows)/len(rows) if rows else None,
        "environment_steps":sum(r["environment_steps"] for r in rows),"astra_calls":sum(r["astra_calls"] for r in rows),
        "note":"Excludes manual, calibration, dry-run and in-progress artifacts"}
    target.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return rows
