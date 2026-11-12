"""
make_dag.py -- write tuning.dag from variants.tsv: python make_dag.py

Per variant: train (or reuse another variant's flow), place, diagnose and
score, the fitting factors in FF_PARTS parallel jobs and their join (ff.sub);
the datasets and isobank's flow (for the fitting-factor shortlist) first, the
plots last.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORY = dict(isobank_flow='long', dataset='short', train='medium', diagnose='short',
                place='medium', ff='medium', plots='express')
FF_PARTS = 3          # per variant: 14 variants x 3 = 42 CPUs, about 1.5 h instead of 4 h


def job_name(step, name):
    """DAG job names: the variant name with '.' and '-' replaced."""
    return f"{step}_{name.replace('.', 'p').replace('-', 'm')}" if name else step


def node(step, name):
    job = job_name(step, name)
    vars_ = f'step="{step}" name="{name or "-"}" category="{CATEGORY[step]}"'
    return job, f'JOB {job} step.sub\nVARS {job} {vars_}\n'


def main():
    with open(os.path.join(HERE, 'variants.tsv')) as fh:
        variants = [l.rstrip('\n').split('\t') for l in fh if l.strip() and not l.startswith('#')]
    names = {v[0] for v in variants}
    jobs, edges = [], []

    jobs.append(node('isobank_flow', None)[1])
    for ds in sorted({v[1] for v in variants}):
        jobs.append(node('dataset', ds)[1])

    leaves = []
    for name, ds, flow, *_ in variants:
        for step in ('train', 'place', 'diagnose'):
            jobs.append(node(step, name)[1])
        j = {step: job_name(step, name) for step in ('train', 'place', 'diagnose', 'ffparts', 'ffjoin')}
        for step, mode, n_jobs, cat in (('ffparts', 'part', FF_PARTS, 'medium'), ('ffjoin', 'join', 1, 'express')):
            jobs.append(f'JOB {j[step]} ff.sub\nVARS {j[step]} mode="{mode}" name="{name}" n_parts="{FF_PARTS}" '
                        f'n_jobs="{n_jobs}" category="{cat}"\n')
        if flow == 'train':
            edges.append(f"PARENT {job_name('dataset', ds)} CHILD {j['train']}")
        elif flow in names:
            edges.append(f"PARENT {job_name('train', flow)} CHILD {j['train']}")
        else:
            raise SystemExit(f'{name}: flow must be "train" or a variant name, not {flow!r}')
        edges += [f"PARENT {j['train']} CHILD {j['place']} {j['diagnose']}",
                  f"PARENT {j['place']} isobank_flow CHILD {j['ffparts']}",
                  f"PARENT {j['ffparts']} CHILD {j['ffjoin']}"]
        leaves += [j['ffjoin'], j['diagnose']]
    jobs.append(node('plots', None)[1])
    edges.append(f'PARENT {" ".join(leaves)} CHILD plots')

    with open(os.path.join(HERE, 'tuning.dag'), 'w') as fh:
        fh.write('# mbank tuning sweep, written by make_dag.py from variants.tsv: do not edit.\n'
                 '# Submit from this folder: condor_submit_dag tuning.dag\n\n')
        fh.write('\n'.join(jobs) + '\n' + '\n'.join(edges) + '\n')
    print(f'tuning.dag: {len(variants)} variants, {len(jobs)} jobs')


if __name__ == '__main__':
    main()
