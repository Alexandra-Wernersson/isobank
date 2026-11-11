"""
diagnose_flow.py -- accuracy of a trained mbank flow, written to a json file.

    python diagnose_flow.py --ini base.ini --flow flow.zip --dataset d15k.dat --out diagnostics.json

On the validation split of the dataset (the last 1 - train-fraction, as
mbank_train_flow), with r = log q_flow + C - log sqrt|det M|:

  * percentiles of |r| and the fraction of points with r < -log(10),
    i.e. whose importance weight exceeds W_max = 10;
  * C - C*, with C* = mean(log sqrt|det M| - log q_flow) the best constant;
  * the fraction of flow samples inside the boundaries, and mbank's log10
    volume (C / ln 10) with the one corrected for the samples outside.
"""
import argparse, json, sys
import numpy as np
import torch

import mbank.parser
from mbank.flow import STD_GW_Flow


def read_ini(ini):
    parser = argparse.ArgumentParser()
    for add in (mbank.parser.add_general_options, mbank.parser.add_metric_options,
                mbank.parser.add_flow_options, mbank.parser.add_range_options):
        add(parser)
    args, _ = parser.parse_known_args([])
    return mbank.parser.updates_args_from_ini(ini, args, parser)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ini', required=True)
    ap.add_argument('--flow', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-samples', type=int, default=100_000)
    a = ap.parse_args()

    args = read_ini(a.ini)
    flow = STD_GW_Flow.load_flow(a.flow)
    data = np.loadtxt(a.dataset)
    theta, ll = data[:, :-1], data[:, -1]
    n_train = int(len(data) * args.train_fraction)

    with torch.no_grad():
        lq = flow.log_prob(torch.tensor(theta, dtype=torch.float32)).numpy()
        samples = flow.sample(a.n_samples).numpy()
    C = float(flow.constant.item())
    r = (lq + C - ll)[n_train:]
    c_best = float(np.mean(ll[:n_train] - lq[:n_train]))
    inside = float(mbank.parser.boundary_keeper(args)(samples, args.variable_format).mean())

    out = dict(abs_r_p50=float(np.median(np.abs(r))), abs_r_p90=float(np.percentile(np.abs(r), 90)),
               abs_r_p99=float(np.percentile(np.abs(r), 99)),
               frac_weight_above_10=float(np.mean(r < -np.log(10))),
               C_minus_Cbest=C - c_best, samples_inside=inside,
               log10_volume=C / np.log(10),
               log10_volume_inside=(C + np.log(max(inside, 1e-300))) / np.log(10))
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    sys.exit(main())
