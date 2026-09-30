"""Read-only viewer of the last committed rollout observation, without simulator access."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np


def latest(root, episode):
    matches=list(root.glob(f'*/task_*/{episode}/metadata.json'))
    if not matches:
        return None
    directory=matches[0].parent
    metadata=json.loads(matches[0].read_text(encoding='utf-8'))
    rows=[]
    for line in (directory/'trajectory.jsonl').read_text(encoding='utf-8').splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    record=next((r for r in reversed(rows) if r.get('next_images')),None)
    if record:
        paths=record['next_images']
        frame=record.get('next_observation_step',record['step']+1)
    else:
        initial=directory/'initial_observation.json'
        if not initial.exists(): return None
        paths=json.loads(initial.read_text(encoding='utf-8'))['images']
        frame=0
    return directory,metadata,record,paths,frame


def build_frame(root,episode):
    result=latest(root,episode)
    if result is None:return None
    directory,metadata,record,paths,frame=result
    panels=[]
    for name in ('primary','wrist'):
        if name not in paths:
            panel=np.zeros((514,480,3),dtype=np.uint8)
            cv2.putText(panel,'WRIST CAMERA NOT AVAILABLE',(20,250),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),1)
            panels.append(panel)
            continue
        # imdecode supports Unicode Windows paths; grayscale/alpha are normalized.
        image=cv2.imdecode(np.frombuffer((directory/paths[name]).read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Incomplete image file')
        h,w=image.shape[:2]
        image=cv2.resize(image,(480,int(h*480/w)),interpolation=cv2.INTER_NEAREST)
        panel=cv2.copyMakeBorder(image,34,0,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
        cv2.putText(panel,name.upper()+' CAMERA',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
        panels.append(panel)
    height=max(p.shape[0] for p in panels)
    panels=[cv2.copyMakeBorder(p,0,height-p.shape[0],0,0,cv2.BORDER_CONSTANT) for p in panels]
    canvas=np.hstack(panels)
    canvas=cv2.copyMakeBorder(canvas,145,62,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
    action=(record or {}).get('semantic_action') or {}
    lines=[f'Observation {frame} | {episode}',
           f"Status from log: {metadata.get('status')} | success: {metadata.get('success',(record or {}).get('success',False))}",
           'Last action: '+' '.join(f'{k}={action.get(k,"-")}' for k in ('precision','x','y','z','gripper'))]
    lines.append('Phase: '+str((record or {}).get('recovery_state','APPROACH'))+' | Grasp: '+str((record or {}).get('grasp_status','NOT_APPLICABLE')))
    for i,line in enumerate(lines):cv2.putText(canvas,line,(12,28+i*30),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(canvas,'Updates after env.step. API waits do not advance simulated time.',(12,canvas.shape[0]-36),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,220,255),1,cv2.LINE_AA)
    cv2.putText(canvas,'Q / Esc: close viewer only. S: save displayed snapshot.',(12,canvas.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,220,255),1,cv2.LINE_AA)
    return canvas,directory,frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='outputs')
    parser.add_argument('--episode',required=True)
    parser.add_argument('--refresh-every-steps',type=int,default=1)
    parser.add_argument('--snapshot',help='Render one image and exit without opening a window')
    args=parser.parse_args()
    if not args.episode.replace('_','').isalnum():parser.error('Invalid episode identifier')
    root=Path(args.root)
    if args.snapshot:
        result=build_frame(root,args.episode)
        if result is None:raise RuntimeError('No committed observation found')
        target=Path(args.snapshot);target.parent.mkdir(parents=True,exist_ok=True)
        cv2.imencode('.png',result[0])[1].tofile(str(target))
        print('Saved observation',result[2],'to',target)
        return
    title='Astra LIBERO - Live Cameras (read only)'
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,960,690)
    refresh=0
    result=None
    shown=-1
    try:
        while True:
            if time.monotonic()>=refresh:
                try:
                    result=build_frame(root,args.episode)
                    if result is not None and (shown<0 or result[2]-shown>=max(1,args.refresh_every_steps) or result[1].parent.parent.name!='in_progress'):
                        cv2.imshow(title,result[0]);shown=result[2]
                except (OSError,ValueError,json.JSONDecodeError):pass
                refresh=time.monotonic()+.5
            key=cv2.waitKey(50)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key==ord('s') and result is not None:
                out=root/'validation'/f'{args.episode}_observation_{result[2]:05d}.png'
                out.parent.mkdir(parents=True,exist_ok=True)
                cv2.imencode('.png',result[0])[1].tofile(str(out))
                print('Saved',out,flush=True)
    finally:cv2.destroyAllWindows()


if __name__=='__main__':
    try: main()
    except cv2.error as exc: print('Camera GUI unavailable; rollout continues:',type(exc).__name__)
