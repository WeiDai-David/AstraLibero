"""Fixed-layout live status GUI. Independent process; no control access."""
import argparse,json,time,textwrap
from pathlib import Path
import cv2
import numpy as np

def render(root,episode):
    files=list(root.glob(f'*/task_*/{episode}/metadata.json'))
    if not files:return None
    p=files[0].parent;m=json.loads(files[0].read_text());live={}
    if (p/'live_status.json').exists():live=json.loads((p/'live_status.json').read_text())
    rows=[]
    for line in (p/'trajectory.jsonl').read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    r=live.get('last_action') or (rows[-1] if rows else {})
    a=r.get('semantic_action',{});state=r.get('next_robot_state',r.get('robot_state',{}))
    attempts=[v for row in rows for v in row.get('attempts',[])]
    lines=[f'Episode: {episode}',f'Task: {m.get("task_instruction", "-")}',
        f'Status: {live.get("episode_status",m.get("status"))} | steps: {live.get("step",0)} | decision: {r.get("decision_id","-")}',
        f'Phase: {live.get("phase",r.get("recovery_state"))} | precision: {a.get("precision","-")}',
        'Semantic: '+json.dumps(a,ensure_ascii=True),
        'Base [m/rad/grip]: '+str(r.get('base_action')),
        'Requested normalized: '+str(r.get('requested_action')),
        'Clipped normalized: '+str(r.get('clipped_action')),
        f'Repeat trans/rot/total: {r.get("translation_repeat")}/{r.get("rotation_repeat")}/{r.get("repeat_count")} | index: {r.get("substep_index")}',
        'Actual cumulative displacement [m]: '+str(r.get('cumulative_eef_displacement')),
        'EEF position: '+str(state.get('position')),'EEF quaternion: '+str(state.get('orientation')),
        'Gripper joints: '+str(state.get('gripper')),
        f'Grasp: {live.get("grasp_status")} | attempt: {live.get("grasp_attempt_id")} | retries: {live.get("grasp_retry_count")}',
        f'Reward: {r.get("reward")} | success: {m.get("success",r.get("success",False))}',
        f'Astra latency: {r.get("astra_latency")} | HTTP attempts: {live.get("astra_calls",len(attempts))} | 429: {max(live.get("http_event",{}).get("rate_limit_count",0),sum(v.get("status_code")==429 for v in attempts))}',
        'API activity: '+str(live.get('http_event',{})), 'Recent actions:']
    for row in rows[-8:]:
        aa=row.get('executed_semantic_action',row.get('semantic_action',{}))
        lines.append(f"step={row.get('step')} decision={row.get('decision_id')} {aa.get('precision')} x={aa.get('x')} y={aa.get('y')} z={aa.get('z')} grip={aa.get('gripper')} success={row.get('success')}")
    expanded=[part for line in lines for part in textwrap.wrap(line,135)]
    canvas=np.full((max(780,len(expanded)*23+35),1180,3),24,dtype=np.uint8)
    for i,line in enumerate(expanded):cv2.putText(canvas,line,(12,25+23*i),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1,cv2.LINE_AA)
    return canvas

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs');p.add_argument('--episode',required=True)
    p.add_argument('--refresh-every-steps',type=int,default=1);p.add_argument('--snapshot');args=p.parse_args()
    if not args.episode.replace('_','').isalnum():p.error('Invalid episode')
    if args.snapshot:
        image=render(Path(args.root),args.episode)
        if image is None:raise RuntimeError('No episode')
        cv2.imencode('.png',image)[1].tofile(args.snapshot);return
    title='Astra LIBERO - Status (read only)';cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1180,850)
    shown_step=-1;shown_status=None
    try:
        while True:
            try:
                image=render(Path(args.root),args.episode)
                if image is not None:
                    files=list(Path(args.root).glob(f'*/task_*/{args.episode}/live_status.json'))
                    live=json.loads(files[0].read_text()) if files else {}
                    n=live.get('step',0);status=(live.get('episode_status'),str(live.get('http_event')))
                    if shown_step<0 or n-shown_step>=max(1,args.refresh_every_steps) or status!=shown_status:
                        cv2.imshow(title,image);shown_step=n;shown_status=status
            except (OSError,ValueError):pass
            if cv2.waitKey(500)&255 in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
    finally:cv2.destroyAllWindows()

if __name__=='__main__':
    try:main()
    except cv2.error as exc:print('Status GUI unavailable; rollout continues:',type(exc).__name__)
