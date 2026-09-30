"""Read-only live rollout monitor. No environment access, reset or API calls."""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


def snapshot(root, episode):
    matches = list(root.glob(f'*/task_*/{episode}/metadata.json'))
    if not matches:
        return ['Waiting for episode directory: ' + episode]
    path = matches[0].parent
    try:
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return ['Metadata updating; waiting for next refresh.']
    rows = []
    trajectory = path / 'trajectory.jsonl'
    try:
        for line in trajectory.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Ignore an incomplete final line during concurrent writing.
    except OSError:
        pass
    executed = [r for r in rows if r.get('execution_valid')]
    latest = executed[-1] if executed else {}
    action = latest.get('semantic_action') or {}
    state = latest.get('next_robot_state') or {}
    attempts = [a for r in rows for a in r.get('attempts', [])]
    age = max(0, time.time() - trajectory.stat().st_mtime) if trajectory.exists() else None
    requests = sorted((path / 'requests').glob('step_*_attempt_*.json'))
    status = meta.get('status', 'unknown')
    output = [
        'ASTRA / LIBERO LIVE MONITOR -- read only',
        'Updated: ' + datetime.now().astimezone().isoformat(timespec='seconds'),
        'Episode: ' + episode,
        'Status (from disk): ' + status,
        'Executed steps: ' + str(len(executed)) + ' | Last executed step (0-based): ' + str(latest.get('step', '-')),
        'Success: ' + str(meta.get('success', latest.get('success', False))),
        'Termination: ' + str(meta.get('termination', '-')),
        'Failure reason: ' + str(meta.get('failure_reason', '-')),
        'Time since last trajectory write: ' + (f'{age:.1f}s' if age is not None else '-'),
        'HTTP attempts logged: ' + str(len(attempts)) + ' | Request files prepared: ' + str(len(requests)),
        'Latest prepared request: ' + (requests[-1].name if requests else '-'),
        'Retryable errors recorded: ' + str(sum(bool(a.get('retryable')) for a in attempts)),
        'NOTE: in_progress is a file status, not a process heartbeat.',
        '      Request files are written before pacing; they do not prove an HTTP call started.',
        '',
        'Latest executed action: ' + ' '.join(f'{k}={action.get(k,"-")}' for k in ('mode','x','y','z','roll','pitch','yaw','gripper')),
        'Model reason: ' + str(action.get('reason', '-')),
        'EEF position: ' + str(state.get('position', '-')),
        'Gripper joints: ' + str(state.get('gripper', '-')),
        'Decision latency: ' + str(latest.get('astra_latency', '-')) + ' seconds',
        '', 'Recent completed actions:',
    ]
    for row in executed[-8:]:
        a = row.get('semantic_action') or {}
        output.append(f"  step={row['step']:4} {a.get('mode','-'):6} x={a.get('x','-'):8} y={a.get('y','-'):8} z={a.get('z','-'):8} gripper={a.get('gripper','-'):5} success={row.get('success')}")
    output += ['', 'Saved data: ' + str(path.resolve()),
               'Refresh: every 2 seconds. Ctrl+C closes ONLY this monitor; rollout keeps running.']
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs')
    parser.add_argument('--episode', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not args.episode.replace('_', '').isalnum():
        parser.error('Invalid episode identifier')
    try:
        while True:
            lines = snapshot(Path(args.root), args.episode)
            if not args.once:
                os.system('cls' if os.name == 'nt' else 'clear')
            print('\n'.join(lines), flush=True)
            if args.once:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print('\nMonitor closed. No commands were sent to the rollout.')


if __name__ == '__main__':
    main()
