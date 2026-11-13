"""
summarise.py -- isobank's run-to-run variance, from the runs of nikhef/retrain.dag.

    python retrain_variance/summarise.py --ref runs/eccentric_5d --runs runs/eccentric_5d_retrain --logs logs

Run 0 is the reference (runs/eccentric_5d: the comparison DAG's flow, placement seed 0). seed<s> retrains the
flow with network seed s (same data, pairs and r_target) and places with seed 0; place<p> re-places the reference
flow with seed p. Prints, and writes to <runs>/summary.md, a markdown table: bank size, coverage (FF >= 0.97) and
its binomial error, 1st percentile and worst FF, and the flow's best held-out p10 of d_z / d_proper; then the mean
and standard deviation over the trainings (run 0 + seeds) and over the placements (run 0 + place).
"""
import argparse, os, re
import numpy as np


def best_p10(log):
    """Highest monitoring p10 in a scripts/train_flow.py log (the checkpoint rule keeps the best)."""
    if not os.path.exists(log):
        return np.nan
    vals = [float(m.group(1)) for m in re.finditer(r'monitor med=[\d.]+ p10=([\d.]+)', open(log).read())]
    return max(vals) if vals else np.nan


def row(name, bank, ff, log):
    n = len(np.load(bank, mmap_mode='r')); f = np.load(ff); p = (f >= .97).mean()
    return dict(name=name, n=n, cov=100 * p, err=100 * np.sqrt(p * (1 - p) / len(f)),
                p1=np.percentile(f, 1), worst=f.min(), p10=best_p10(log), n_inj=len(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--logs', required=True)
    args = ap.parse_args()
    rows = [row('reference (run 0)', os.path.join(args.ref, 'bank_isobank.npy'),
                os.path.join(args.ref, 'ff_isobank.npy'), os.path.join(args.logs, 'train_isobank.out'))]
    for run in sorted(os.listdir(args.runs)):
        d = os.path.join(args.runs, run)
        if os.path.exists(os.path.join(d, 'ff.npy')):
            rows.append(row(run, os.path.join(d, 'bank.npy'), os.path.join(d, 'ff.npy'),
                            os.path.join(args.logs, f'retrain_train_{run}_0.out')))
    lines = ['| Run | Templates | Coverage, FF ≥ 0.97 | 1st pct FF | Worst FF | Flow p10 (held-out) |',
             '|---|---|---|---|---|---|']
    for r in rows:
        p10 = '–' if np.isnan(r['p10']) else f"{r['p10']:.3f}"      # no training log for a re-placement
        lines.append(f"| {r['name']} | {r['n']:,} | {r['cov']:.2f} ± {r['err']:.2f} % | {r['p1']:.4f} "
                     f"| {r['worst']:.4f} | {p10} |")
    for label, keep in (('trainings (run 0 + seed*)', lambda r: not r['name'].startswith('place')),
                        ('placements (run 0 + place*)', lambda r: not r['name'].startswith('seed'))):
        sel = [r for r in rows if keep(r)]
        if len(sel) > 1:
            n, c = np.array([r['n'] for r in sel]), np.array([r['cov'] for r in sel])
            lines.append(f"| **{label}, mean ± std** | {n.mean() / 1e6:.3f}M ± {n.std(ddof=1) / 1e6:.3f}M "
                         f"| {c.mean():.2f} ± {c.std(ddof=1):.2f} % | | | |")
    text = f"isobank, eccentric 5D, {rows[0]['n_inj']:,} injections of seed 4002, radius = r_target.\n\n" \
           + '\n'.join(lines) + '\n'
    with open(os.path.join(args.runs, 'summary.md'), 'w') as fh:
        fh.write(text)
    print(text + f"-> {os.path.join(args.runs, 'summary.md')}")


if __name__ == '__main__':
    main()
