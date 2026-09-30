import argparse
from src.trajectory.metrics import analyze
p=argparse.ArgumentParser();p.add_argument('--root',default='outputs');p.add_argument('--output',default='results.csv'); a=p.parse_args(); print(f'episodes: {len(analyze(a.root,a.output))}')
