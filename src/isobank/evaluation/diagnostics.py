"""
isobank.diagnostics
===================

Diagnostic and summary plots: flow isometry quality (independent of the
training pairs), r_target calibration curves, FF survival functions and
coverage maps.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

__all__ = ['plot_calibration', 'plot_isometry_diagnostic', 'plot_coverage',
           'plot_bank_scatter', 'plot_ff_distribution', 'folding_check']

####################################################################################################################

def plot_calibration(diag, r_target, mu=0.03, title='', savefile=None):
	"""
	Calibration curve: TRUE mismatch vs proper distance with the calibrated
	r_target marked. ``diag`` is the diagnostics dict from
	:func:`isobank.calibration.calibrate_r_target`.
	"""
	fig, ax = plt.subplots(figsize=(7, 4.5))
	ax.scatter(diag['d_proper'], diag['mismatches'], s=3, alpha=0.15,
	           color='#16a34a', rasterized=True)
	ax.plot(diag['bin_dist'], diag['bin_mean'], 'o-', color='#16a34a', ms=5, lw=2)
	ax.axhline(mu, color='k', lw=1.5, ls='--', label=f'mu = {mu}')
	ax.axvline(r_target, color='#16a34a', lw=1.5, ls=':',
	           label=f'r_target = {r_target:.4f}')
	ax.axvline(diag['r_metric'], color='#64748b', lw=1.0, ls=':',
	           label=f'r_metric = {diag["r_metric"]:.4f}')
	ax.set_xlabel(r'Proper distance $d_{\rm proper}$')
	ax.set_ylabel(r'TRUE mismatch $1 - \mathcal{M}$')
	ax.set_title(title or 'r_target calibration')
	ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
	ax.set_ylim(-0.005, min(0.4, diag['mismatches'].max() * 1.1))
	plt.tight_layout()
	if savefile:
		plt.savefig(savefile, dpi=150, bbox_inches='tight', facecolor='white')
		print(f"Saved: {savefile}")
	return fig

####################################################################################################################

def plot_isometry_diagnostic(flow, metric, pairs, embed_fn=None,
                             title='', savefile=None):
	"""
	Independent isometry check on fresh pairs: ratio histogram and
	d_proper vs d_z scatter.

	Parameters
	----------
		pairs: tuple
			``(t1, t2, d_true)`` fresh (non-training) pairs
	"""
	if embed_fn is None: embed_fn = lambda t: t
	t1, t2, d_true = pairs
	with torch.no_grad():
		z1, _ = flow(embed_fn(t1)); z2, _ = flow(embed_fn(t2))
		d_z = torch.norm(z1 - z2, dim=1).numpy()
	d_true = d_true.numpy() if isinstance(d_true, torch.Tensor) else d_true
	ratios = d_z / (d_true + 1e-8)

	p10, med, p90 = (np.percentile(ratios, 10), np.median(ratios),
	                 np.percentile(ratios, 90))

	fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
	ax = axes[0]
	ax.hist(ratios, bins=60, color='#0284c7', alpha=0.8)
	for v, l, c in [(p10, f'p10={p10:.3f}', '#dc2626'),
	                (med, f'median={med:.3f}', 'k'),
	                (p90, f'p90={p90:.3f}', '#dc2626')]:
		ax.axvline(v, color=c, lw=1.5, ls='--', label=l)
	ax.axvline(1.0, color='#16a34a', lw=1.5)
	ax.set_xlabel(r'$d_z / d_{\rm proper}$'); ax.set_ylabel('count')
	ax.set_title('Isometry ratio'); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

	ax = axes[1]
	ax.scatter(d_true, d_z, s=4, alpha=0.25, color='#0284c7', rasterized=True)
	lim = max(d_true.max(), d_z.max()) * 1.05
	ax.plot([0, lim], [0, lim], color='#16a34a', lw=1.5)
	ax.set_xlabel(r'$d_{\rm proper}$'); ax.set_ylabel(r'$d_z$')
	ax.set_title('Proper vs latent distance'); ax.grid(True, alpha=0.4)

	plt.suptitle(title, fontsize=11)
	plt.tight_layout()
	if savefile:
		plt.savefig(savefile, dpi=150, bbox_inches='tight', facecolor='white')
		print(f"Saved: {savefile}")
	print(f"Diagnostic: p10={p10:.4f}  median={med:.4f}  p90={p90:.4f}")
	return fig

####################################################################################################################

def plot_coverage(best_matches, signals, mu=0.03, mass_col=0, log_mass=True,
                  y_col=1, y_label='q', title='', savefile=None):
	"""
	FF survival function and coverage map coloured by fitting factor.
	"""
	thr   = 1.0 - mu
	cov   = (best_matches >= thr).mean() * 100
	fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

	ax = axes[0]
	ff_vals = np.linspace(min(0.9, best_matches.min()), 1.001, 500)
	surv = np.array([(best_matches >= v).mean() * 100 for v in ff_vals])
	ax.plot(ff_vals, surv, color='#16a34a', lw=2)
	ax.axvline(thr, color='#dc2626', lw=1.5, ls='--', label=f'FF={thr}')
	ax.set_xlabel('Fitting Factor')
	ax.set_ylabel('Fraction with FF >= threshold (%)')
	ax.set_title(f'FF survival — {cov:.1f}% at FF>={thr}')
	ax.legend(); ax.grid(True, alpha=0.4); ax.set_ylim(0, 103)

	ax = axes[1]
	x = 10**signals[:, mass_col] if log_mass else signals[:, mass_col]
	sc = ax.scatter(x, signals[:, y_col], s=6, c=best_matches, cmap='RdYlGn',
	                vmin=min(0.92, best_matches.min()), vmax=1.0, rasterized=True)
	plt.colorbar(sc, ax=ax, label='Fitting Factor')
	ax.set_xlabel(r'Total mass $M\;[M_\odot]$' if log_mass else 'mass')
	ax.set_ylabel(y_label)
	if log_mass: ax.set_xscale('log')
	ax.set_title('Coverage map'); ax.grid(True, alpha=0.3)

	plt.suptitle(title, fontsize=11)
	plt.tight_layout()
	if savefile:
		plt.savefig(savefile, dpi=150, bbox_inches='tight', facecolor='white')
		print(f"Saved: {savefile}")
	return fig

####################################################################################################################

def plot_bank_scatter(templates, mass_col=0, log_mass=True, y_col=1,
                      y_label='q', title='', savefile=None):
	"""Simple bank scatter in the mass plane."""
	fig, ax = plt.subplots(figsize=(7, 5))
	x = 10**templates[:, mass_col] if log_mass else templates[:, mass_col]
	ax.scatter(x, templates[:, y_col], s=1, alpha=0.3, color='#16a34a',
	           rasterized=True)
	ax.set_xlabel(r'Total mass $M\;[M_\odot]$' if log_mass else 'mass')
	ax.set_ylabel(y_label)
	if log_mass: ax.set_xscale('log')
	ax.set_title(title or f'{len(templates):,} templates')
	ax.grid(True, alpha=0.3)
	plt.tight_layout()
	if savefile:
		plt.savefig(savefile, dpi=150, bbox_inches='tight', facecolor='white')
		print(f"Saved: {savefile}")
	return fig

####################################################################################################################

def plot_ff_distribution(best_matches, mu=0.03, title='', savefile=None):
	"""
	The standard template-bank paper figure: fitting-factor histogram
	(log-y) and cumulative FF distribution, with the minimal-match line.
	"""
	thr    = 1.0 - mu
	n      = len(best_matches)
	cov    = (best_matches >= thr).mean() * 100
	med    = np.median(best_matches)

	fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

	ax = axes[0]
	bins = np.linspace(max(0.85, best_matches.min() - 0.01), 1.0, 60)
	ax.hist(best_matches, bins=bins, color='#0284c7', alpha=0.85)
	ax.axvline(thr, color='#dc2626', lw=1.8, ls='--', label=f'FF = {thr}')
	ax.axvline(med, color='k', lw=1.5, ls=':', label=f'median = {med:.3f}')
	ax.set_yscale('log')
	ax.set_xlabel('Fitting factor'); ax.set_ylabel('Number of injections')
	ax.set_title('FF distribution')
	ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

	ax = axes[1]
	ff_sorted = np.sort(best_matches)
	cdf = np.arange(1, n + 1) / n * 100
	ax.plot(ff_sorted, cdf, color='#0284c7', lw=2)
	ax.axvline(thr, color='#dc2626', lw=1.8, ls='--')
	frac_below = (best_matches < thr).mean() * 100
	ax.axhline(frac_below, color='#dc2626', lw=1.0, ls=':', alpha=0.7)
	ax.annotate(f'{frac_below:.1f}% below FF={thr}\n(coverage {cov:.1f}%)',
	            xy=(thr, frac_below),
	            xytext=(bins[0] + 0.01, min(frac_below + 25, 80)),
	            color='#dc2626', fontsize=9,
	            arrowprops=dict(arrowstyle='->', color='#dc2626'))
	ax.set_xlabel('Fitting factor')
	ax.set_ylabel('Cumulative fraction of injections (%)')
	ax.set_title('Cumulative FF distribution')
	ax.grid(True, alpha=0.4)
	ax.set_xlim(bins[0], 1.0); ax.set_ylim(0, 102)

	plt.suptitle(title, fontsize=11)
	plt.tight_layout()
	if savefile:
		plt.savefig(savefile, dpi=150, bbox_inches='tight', facecolor='white')
		print(f"Saved: {savefile}")
	return fig


####################################################################################################################

def folding_check(flow, metric, templates, lo, hi, embed_fn=None, valid_fn=None,
                  n_inj=800, n_true=600, seed=99, fold_factor=3.0, verbose=True):
	"""
	Test an isometric flow for GLOBAL FOLDING.

	A flow can be near-perfectly isometric on LOCAL pairs and still map
	physically distant regions on top of each other. The p10 diagnostic used
	for checkpoint selection is measured on pairs at 0.5-1.5 r_target, so it
	is structurally blind to this failure: it never compares points that the
	flow has wrongly brought together.

	Folding breaks the pipeline in two ways at once. Placement rejects
	candidates that sit near a template's folded image despite being
	physically far from it, so the bank is left with real holes; and k-NN
	retrieval returns templates that are nowhere near the signal, so coverage
	collapses. The tell-tale signature is coverage that is FLAT in the latent
	distance to the nearest template — once the flow folds, that distance has
	stopped carrying information.

	This check measures the ratio for the pairs that actually matter: each
	injection and the template the flow retrieves as its nearest neighbour.
	Contrast with a local ratio test on the SAME flow (aligned-spin wide,
	M in [2, 400]): local pairs gave d_z/d_proper = 1.0008 at the median,
	while retrieved nearest neighbours gave 0.139 — the retrieved template
	was typically 7x farther away in proper distance than the latent
	distance claimed, for 69% of injections.

	Returns
	-------
		diag: dict
			ratio, d_z, d_proper, mismatch, folded_frac
	"""
	import numpy as np, torch
	from scipy.spatial import cKDTree

	if embed_fn is None: embed_fn = lambda t: t
	D   = len(lo)
	rng = np.random.default_rng(seed)

	def _z(a):
		out = []
		for i in range(0, len(a), 200_000):
			with torch.no_grad():
				z, _ = flow(embed_fn(torch.from_numpy(a[i:i+200_000].astype(np.float32))))
			out.append(z.numpy())
		return np.vstack(out)

	inj, have = [], 0
	while have < n_inj:
		c = rng.uniform(lo, hi, (n_inj, D)).astype(np.float32)
		if valid_fn is not None: c = c[valid_fn(c)]
		inj.extend(c); have = len(inj)
	inj = np.array(inj[:n_inj])

	tree    = cKDTree(_z(np.asarray(templates, dtype=np.float32)))
	d_z, ix = tree.query(_z(inj), k=1, workers=-1)
	nn      = np.asarray(templates, dtype=np.float32)[ix]

	S  = min(n_true, len(inj))
	t1 = torch.from_numpy(inj[:S]); t2 = torch.from_numpy(nn[:S])
	with torch.no_grad():
		d_proper = metric.proper_distance(t1, t2).numpy()
		mismatch = 1.0 - np.asarray(metric.match(t1, t2))
	d_z   = d_z[:S]
	ratio = d_z / np.maximum(d_proper, 1e-12)
	folded_frac = float((d_proper > fold_factor * np.maximum(d_z, 1e-6)).mean())

	if verbose:
		print("Folding check (injection vs its RETRIEVED nearest template):")
		print(f"  median d_z            : {np.median(d_z):.4f}")
		print(f"  median d_proper       : {np.median(d_proper):.4f}")
		print(f"  ratio d_z/d_proper    : p10={np.percentile(ratio,10):.3f}  "
		      f"p50={np.median(ratio):.3f}  p90={np.percentile(ratio,90):.3f}")
		print(f"  folded (d_proper > {fold_factor:g}x d_z) : {100*folded_frac:.1f}%")
		print(f"  median mismatch to NN : {np.median(mismatch):.5f}")
		if folded_frac > 0.10:
			print("  WARNING: the flow is globally folded. Latent distance does not "
			      "identify nearby templates,\n           so both placement and k-NN "
			      "retrieval are unreliable regardless of the placement radius.")
	return dict(ratio=ratio, d_z=d_z, d_proper=d_proper,
	            mismatch=mismatch, folded_frac=folded_frac)
