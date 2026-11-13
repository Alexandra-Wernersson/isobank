"""
Train one ablation arm:   python train_arm.py <arm> <seed>   (or: <array index>)

Calls isobank.train_isometric_flow unchanged. A switched-off term gets weight
0 but its pair set is still passed, so logging and the checkpoint criterion
(max min(p10_monitor, p10_edge)) are identical in every arm.
"""
import sys, json, time
import numpy as np, torch
from common import *
from isobank import train_isometric_flow

def main():
	pos = [a for a in sys.argv[1:] if '=' not in a]; kv = parse_kv(sys.argv[1:])
	p = int(kv.get('p', 0))
	if len(pos) == 1:                                        # slurm array index
		i = int(pos[0]); arm, seed = ARM_ORDER[i % len(ARMS)], SEEDS[i // len(ARMS)]
	else:
		arm, seed = pos[0], int(pos[1])
	weights, tail_w = {**ARMS, **EXTRA_ARMS}[arm]
	out = run_dir(arm, seed, p)
	torch.manual_seed(seed); np.random.seed(seed)
	print(f'=== arm {arm}  seed {seed}  pair seed {p}  weights {weights}  tail_w {tail_w} ==='); sys.stdout.flush()

	C = load_config(); metric = C.make_metric()
	P = torch.load(pairs_path(p), weights_only=False)
	t0 = time.time()
	flow, hist = train_isometric_flow(
		metric, P['r_target'], C.variable_format,
		P['long'], P['local'], P['bound'], P['monitor'],
		d_flow=C.d_flow, n_layers=C.n_layers, hidden_dim=C.hidden_dim,
		masks=C.masks, depth=C.DEPTH, edge_pairs=P['edge'],
		weights=weights, tail_w=tail_w, embed_fn=C.embed_fn,
		train_data=P['train_data'], phi_col=C.phi_col, n_epochs=C.N_EPOCHS,
		best_path=os.path.join(out, 'flow_best.pt'),
		final_path=os.path.join(out, 'flow_final.pt'),
		extra_ckpt={'embedded': False, 'arm': arm, 'seed': seed, 'pair_seed': p})
	hist['train_minutes'] = (time.time() - t0) / 60
	with open(os.path.join(out, 'history.json'), 'w') as f:
		json.dump(hist, f)

if __name__ == '__main__':
	main()
