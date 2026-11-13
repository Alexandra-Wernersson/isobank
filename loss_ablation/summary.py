"""
Rebuild the numbers of the ablation section (retraining variance, loss ablation, weight scan, aligned-spin test)
from the saved results.

  python summary.py                 # reads results/ (our runs)
  python summary.py --live          # reads ecc5d/runs and al3d/runs instead (your reruns)
"""
import argparse, glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ECC_GROUPS = [
	('Full objective (retraining variance)', ['full']),
	('Ablation', ['no_global', 'no_local', 'no_bound', 'no_edge', 'no_tail', 'loc_edge', 'global_only']),
	('Weight scan', ['loc_x0p5', 'loc_x2', 'bnd_x0p5', 'bnd_x2', 'edge_x0p5', 'edge_x2', 'tail_1', 'tail_10']),
]


def arm_of(name):
	return re.sub(r'_(p\d+_)?s\d+$', '', name)


def load(pattern):
	out = {}
	for f in sorted(glob.glob(pattern)):
		d = json.load(open(f)); out.setdefault(arm_of(os.path.basename(os.path.dirname(f))), []).append(d)
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--live', action='store_true')
	a = ap.parse_args()
	e5 = os.path.join(HERE, 'ecc5d', 'runs') if a.live else os.path.join(HERE, 'results', 'ecc5d')
	a3 = os.path.join(HERE, 'al3d', 'runs') if a.live else os.path.join(HERE, 'results', 'al3d')

	R = load(os.path.join(e5, '*', 'matched_final.json'))
	ref = np.array([100 * d['miss'] for d in R.get('full', [])])
	print('Eccentric 5D: banks of ~2.27M templates, 4,000 injections; miss rate [%]')
	if len(ref): print(f'  full objective: mean {ref.mean():.2f}, sd {ref.std(ddof=1):.2f}, range {ref.min():.2f}-{ref.max():.2f} over {len(ref)} retrainings')
	for title, arms in ECC_GROUPS:
		print(f'\n  {title}')
		for arm in arms:
			if arm not in R: continue
			m = np.array([100 * d['miss'] for d in R[arm]]); r = [d['radius_factor'] for d in R[arm]]; n = [d['n_templates'] for d in R[arm]]
			diff = f'{m.mean() - ref.mean():+.2f}' if len(ref) and arm != 'full' else '--'
			print(f'    {arm:12s} n={len(m)}  miss {", ".join(f"{x:.2f}" for x in m):34s} mean {m.mean():.2f}  diff {diff:>6}  '
			      f'radius {min(r):.2f}-{max(r):.2f} r_target  templates {min(n):,}-{max(n):,}')

	A = load(os.path.join(a3, '*', 'matched.json'))
	if A:
		print('\nAligned-spin 3D: banks of ~23,800 templates, 5,000 injections; miss rate [%]  (paper_full = flow of the main text)')
		for key, arms in [('full', ['full', 'paper_full']), ('no_global', ['no_global'])]:
			rows = [d for arm in arms for d in A.get(arm, [])]
			m = np.array([100 * d['miss'] for d in rows]); r = [d['radius_factor'] for d in rows]
			print(f'    {key:12s} n={len(m)}  miss {", ".join(f"{x:.2f}" for x in m):30s} mean {m.mean():.2f}  radius {min(r):.2f}-{max(r):.2f}')


if __name__ == '__main__':
	main()
