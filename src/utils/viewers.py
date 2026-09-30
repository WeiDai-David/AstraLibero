"""Read-only subprocess viewers, started once per episode and never tied to control lifetime."""
import os
import subprocess
import sys
from pathlib import Path


def start_viewers(logger, config=None):
    cfg={"enabled":True,"show_status_window":True,"show_camera_window":True,"refresh_every_steps":1,**(config or {})}
    if not cfg['enabled']:return []
    if os.name!='nt' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        logger.metadata['viewer_status']='headless_fallback';return []
    project=Path(__file__).resolve().parents[2]
    launched=[]
    logdir=logger.root/'viewer_logs'/logger.dir.name
    logdir.mkdir(parents=True,exist_ok=True)
    for key,script in [('show_status_window','status_viewer.py'),('show_camera_window','view_rollout.py')]:
        if not cfg.get(key):continue
        cmd=[sys.executable,str(project/'scripts'/script),'--root',str(logger.root),
             '--episode',logger.dir.name,'--refresh-every-steps',str(cfg['refresh_every_steps'])]
        try:
            with (logdir/(script+'.log')).open('ab') as log:
                flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
                p=subprocess.Popen(cmd,cwd=project,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
            launched.append({'script':script,'pid':p.pid})
        except OSError as exc:
            logger.metadata.setdefault('viewer_warnings',[]).append(type(exc).__name__)
    logger.metadata['viewer_processes']=launched
    logger.metadata['viewer_status']='independent; GUI failure does not stop rollout'
    logger._write_metadata()
    return launched
