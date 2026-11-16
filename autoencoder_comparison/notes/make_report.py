import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.utils import ImageReader

S, OUT = sys.argv[1], sys.argv[2]
MUTED = '#6b7280'
st = getSampleStyleSheet()
body = ParagraphStyle('b', parent=st['BodyText'], fontSize=10.5, leading=14.5, spaceAfter=6)
h1 = ParagraphStyle('h1', parent=st['Title'], fontSize=17, spaceAfter=4)
h2 = ParagraphStyle('h2', parent=st['Heading2'], fontSize=12.5, spaceBefore=9, spaceAfter=4)
h3 = ParagraphStyle('h3', parent=st['Heading3'], fontSize=11, spaceBefore=7, spaceAfter=3)
small = ParagraphStyle('s', parent=body, fontSize=9, leading=12, textColor=colors.HexColor(MUTED))
cell = ParagraphStyle('c', parent=body, fontSize=8.8, leading=11.2, spaceAfter=0)
P = lambda t, s=body: Paragraph(t, s)
li = ParagraphStyle('li', parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
bullet = lambda t: Paragraph(t, li, bulletText='•')
code = lambda t: f'<font face="Courier">{t}</font>'


def table(rows, widths, bold_last=False):
    rows = [[Paragraph(str(c), cell) for c in r] for r in rows]
    tb = Table(rows, colWidths=widths, repeatRows=1)
    style = [('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.HexColor(MUTED)),
             ('LINEBELOW', (0, -1), (-1, -1), 0.6, colors.HexColor(MUTED)),
             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]
    if bold_last:
        style.append(('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f1f5f9')))
    tb.setStyle(TableStyle(style))
    return tb


def figure(path, width):
    w, h = ImageReader(path).getSize()
    return Image(path, width=width, height=width * h / w)


B = lambda t: f'<b>{t}</b>'
story = [
    P('Baselines for the reviewer: autoencoders and a tuned mbank', h1),
    P('isobank reviewer response. isobank\'s numbers are the paper\'s bank (2.27M templates, 96.6 % coverage) '
      'unless stated, in the text and the figures. Aligned-spin 3D numbers from the paper and Alexandra\'s SVD tests; 13 November 2026.',
      small),

    P('In short', h2),
    bullet('<b>Cabass et al. (2025)</b> (SVD coefficients, then an autoencoder, then a grid) works only when the '
           'bank is split into enough sub-banks: 1 % coverage with one bank, 80 % with 10, 97.2 % with 20 (3.41M '
           'templates, at the lattice spacing designed for a match of 0.97), against 96.75 % for the paper\'s isobank '
           'bank with 2.27M on the same injections. It needs 51 % more templates for 0.4 points more coverage, with '
           'a much longer tail of badly covered signals.'),
    bullet('<b>Its templates are approximations</b>: waveforms rebuilt from a few SVD coefficients with one fixed '
           'amplitude shape, not exact waveforms at physical parameters, as isobank and mbank produce. A plain grid '
           'in the same coefficients does as well as isobank on the eccentric space (99.7 %), but on the aligned-spin '
           'space, where the amplitude shape varies, it covers 93 % at isobank\'s bank size against 99.7 %.'),
    bullet(f'{B("An autoencoder trained on the physical parameters")} with a local metric loss does not learn the '
           'distances: it either folds the space (18 % coverage) or stretches it (99 % coverage, but 31.8M templates, '
           'ten times isobank).'),
    bullet(f'{B("mbank, retrained with 11 different settings")}, stays at 93.4–94.8 % with about 3M templates, inside '
           'the scatter of three repeats with the paper\'s settings. The paper\'s isobank bank covers 96.6 % with '
           '2.27M templates, 23 % fewer. Tuning does not close the gap.'),
    bullet(f'{B("Compute.")} isobank takes 3.7 CPU-hours to construct its bank, mbank 1 to 2, Cabass et al. '
           'about 125: all less than searching one hour of data (about 1,200 CPU-hours for 3M templates, in PyCBC\'s '
           'convention). The search cost grows with the number of templates; the paper\'s isobank bank has 23 % fewer '
           'than mbank\'s, at higher coverage.'),

    P('1. The question and what we ran', h2),
    P('The reviewer asked whether isobank\'s multi-scale loss is better than metric-preserving autoencoders, and '
      'whether mbank could do better with tuning. We tested three things on the eccentric 5D space (TaylorF2Ecc, '
      'O3 noise), with the same injections as the paper:'),
    table([['test', 'what it is', 'injections'],
           ['Cabass et al. (2025)', 'each waveform is described by a few SVD coefficients; a small network '
                                    'compresses them, and templates sit on a grid in the compressed space',
            'first 4,000 of seed 4002'],
           ['autoencoder on parameters', 'the same kind of network, trained directly on the masses, spins and '
                                         'eccentricity', 'the same 4,000'],
           ['mbank sweep', 'mbank retrained 14 times, changing one setting at a time', 'all 10,000 of seed 4002']],
          [3.6 * cm, 9.6 * cm, 3.3 * cm]),
    Spacer(1, 4),
    P('Coverage is the fraction of injected signals that some template matches to FF >= 0.97. Cabass et al.\'s method is scored with '
      'Alexandra\'s overlap (template rebuilt from its coefficients, no extra time and phase search), which is '
      'slightly pessimistic; isobank, the parameter autoencoders and mbank use the true match.', small),

    KeepTogether([P('2. Cabass et al. (2025)', h2),
    table([['bank', 'templates', 'coverage', '1st pct FF', 'lowest FF'],
           ['isobank, paper bank', '2.27M', '96.75 +/- 0.28 %', '0.965', '0.935'],
           ['SVD grid, no autoencoder', '3.33M', '99.70 +/- 0.09 %', '0.974', '0.822'],
           ['Cabass et al. (2025), 20 sub-banks', '3.41M', '97.17 +/- 0.26 %', '0.954', '0.618'],
           ['Cabass et al. (2025), 10 sub-banks', '3.28M', '79.77 +/- 0.64 %', '0.837', '0.730'],
           ['Cabass et al. (2025), one bank', '10k', '1.27 %', '< 0', '< 0']],
          [6.2 * cm, 2.2 * cm, 3.3 * cm, 2.4 * cm, 2.4 * cm]),
    ]),
    Spacer(1, 6),
    KeepTogether([
        figure(f'{S}/rep_cdf.png', 12.2 * cm),
        P('Figure 1. Cumulative distribution of fitting factors: the height of each curve at the dotted line is the '
          'fraction of signals missed. isobank: the paper\'s bank (2.27M). The SVD grid and both Cabass et al. banks '
          'are at lattice spacing 0.44, whose covering radius gives a worst-case match of 0.97. With 20 sub-banks '
          'Cabass et al. cover 0.4 points more than isobank with 51 % more templates, and have a much longer tail '
          '(lowest fitting factor 0.62 against 0.94).',
          small)]),
    P(f'{B("Approximate templates.")} The method describes each waveform by a few SVD coefficients of its phase, '
      'with one amplitude shape for the whole bank, and its templates are waveforms rebuilt from those coefficients. '
      'They are approximations of the waveform model, not the exact waveform at a physical point. isobank and mbank '
      'place physical parameters and generate the exact waveform there; that is the point of learning the metric '
      'with a flow or an autoencoder in the first place. The SVD grid (Alexandra\'s test: the same coefficients, no '
      'network) shows what the representation alone gives: 99.7 % with 3.33M templates, with the same approximate '
      'templates. On this space the autoencoder adds nothing to it, only reconstruction error.'),
    P(f'{B("Why the autoencoder needs sub-banks.")} The first coefficient spans a range hundreds of times larger than '
      'the second and third, and the network must reproduce all three to about 0.1. Trained on the whole space it '
      'cannot: its errors are about 1 to 2, and its templates are essentially random. Split into slices of the first '
      'coefficient, each with its own network (as Cabass et al. do), each network covers a narrow range. The more '
      'slices, the better: 1 %, 80 %, 96.6 % for 1, 10 and 20. With 20, encoding and decoding alone already loses '
      '1.5 % of the signals, before any grid. With 2 compressed dimensions (Cabass\'s choice) coverage stays near '
      '54 %: two numbers cannot hold three coefficients.'),
    P(f'{B("How close we are to their code.")} Same input (10 SVD coefficients), network (MLP, 64 hidden units, 4 '
      'layers), loss (reconstruction plus pairwise distances), optimiser and training length as their code; we use '
      'double precision, which can only help. One difference remains: their sub-banks each have their own SVD, ours '
      'share Alexandra\'s and are slices of it.'),

    KeepTogether([
        P(f'{B("The aligned-spin space.")} Alexandra also ran the SVD grid on the 3D aligned-spin space (IMRPhenomD, '
          '30–50 solar masses, first 5,000 injections of seed 4242). There the amplitude shape changes across the '
          'space, and one shape for the whole bank caps the match: the amplitude alone already costs up to 2.5 %.'),
        table([['bank', 'templates', 'coverage'],
               ['isobank', '23,810', '99.72 %'],
               ['mbank, stochastic placement (paper)', '23,167', '99.80 %'],
               ['SVD grid, spacing 0.283', '21,283', '93.1 %'],
               ['SVD grid, spacing 0.20', '43,884', '98.0 %'],
               ['SVD grid, spacing 0.15', '79,862', '98.7 %']],
              [8.5 * cm, 3.0 * cm, 3.0 * cm]),
        P('isobank and mbank: all 10,000 injections of seed 4242 (the paper); SVD grid: the first 5,000 of them, '
          'real signal amplitudes, bank size from counted grid cells.', small)]),
    Spacer(1, 4),
    P('At isobank\'s bank size the SVD grid covers 93 %, and even with 3.4 times more templates it stops near 98.7 %. '
      'We did not run Cabass et al.\'s autoencoder here: on the eccentric space it only lost coverage against the SVD '
      'grid it is built on, so with one bank it cannot do better than these numbers. Their own GW banks avoid the '
      'problem by splitting the space by hand into several banks with different amplitude shapes; isobank needs no '
      'such split. The numbers in their paper are for those banks, not for this space, so they cannot be compared '
      'directly.'),

    KeepTogether([P('3. Autoencoder on the physical parameters', h2),
    table([['loss', 'templates', 'coverage', 'what goes wrong'],
           ['encoder loss', '42k', '17.7 %', 'folds: far-apart waveforms land next to each other, so one template '
                                             'is placed where two are needed'],
           ['decoder loss (Cabass, App. B)', '31.8M', '98.8 %', 'stretches: distances come out up to about 3 times '
                                                               'too large, so ten times too many templates are placed']],
          [4.2 * cm, 2.0 * cm, 2.0 * cm, 8.3 * cm])]),
    Spacer(1, 4),
    P('Both losses only check distances between infinitely close points, on a network that can map two places to '
      'one. isobank differs in three ways, and each one addresses one of these failures: its map is invertible by '
      'construction (it cannot fold), its loss checks the map that is used for placement (the decoder loss checks '
      'the reverse one, which is why its latent space stretches), and its loss compares distances at the template '
      'spacing and between far-apart points, not just infinitesimally.'),

    KeepTogether([
        P('4. mbank: retrained with other settings, and a minimal-match sweep', h2),
        P('Two reviewers asked whether mbank was given a fair chance. Reviewer 6xqS asked which of mbank\'s settings '
          'can be tuned and how sensitive it is to them (4a); reviewer 6RHG asked for a sweep of mbank\'s minimal '
          'match before attributing the smaller bank to isobank\'s learned coordinates (4b). Both sweeps score the '
          'banks as in the paper: 10,000 injections of seed 4002, each compared with its 500 nearest templates, a '
          'signal being covered when its fitting factor is at least 0.97. Unlike the loss ablation (section 6), the '
          'bank size is not fixed: it is whatever mbank\'s placement produces, and is reported next to the coverage.'),
        P('4a. mbank retrained with other settings (reviewer 6xqS)', h3),
        P('Each variant retrains mbank\'s normalizing flow with one change from the paper\'s settings, then places a '
          'bank with the paper\'s placement (random, minimal match 0.97, covering fraction 0.9). What was changed:'),
    ]),
    bullet(f'{B("Repeats")} (columns run 1-3, grey; nothing changed): the run-to-run scatter of mbank with the paper\'s settings.'),
    bullet(f'{B("Loss weighting")} (columns p = 0.5, 1; unweighted by default, here weights (l - min l)<super>p</super>): the flow fits the '
           'log metric volume l at the training points; the weights put more emphasis on the dense regions, where '
           'most templates go.'),
    bullet(f'{B("Starting value of the constant C")} (columns q = 0.5, 0.99; default q = 0.9): the flow reproduces l up to an '
           'additive constant C, learned in training; this sets its first value, as a quantile of the training l.'),
    bullet(f'{B("Margin around the parameter box")} (columns m = 0.001, 0.05; default m = 0.01 per side): how far the flow\'s range '
           'extends beyond the boundaries of the parameter space.'),
    bullet(f'{B("Learning rate")} (column lr = 5e-4; default 5e-3) and {B("batch size")} (column batch = 1k; default 5,000): the step size of the '
           'optimizer and the number of training points per step.'),
    bullet(f'{B("Training points")} (column N = 50k; default N = 15,000): the number of points at which the metric is computed to '
           'train the flow.'),
    bullet(f'{B("Placement by importance sampling")} (columns W<sub>max</sub> = 10, '
           'infinity; random placement by default, here importance sampling with weights clipped at W<sub>max</sub>): how the covering fraction is estimated while placing, with the flow '
           'of the first repeat.'),
    KeepTogether([
        figure(f'{S}/rep_violins.png', 16.3 * cm),
        P('Figure 2. One column per bank; colour shows what was changed from mbank\'s default settings, and the '
          'label under each column the new value. Defaults (grey, three repeats): unweighted loss, q = 0.9, '
          'm = 0.01, learning rate 5e-3, batch 5,000, N = 15,000 training points, random placement. Top: '
          'coverage (signals with fitting factor above 0.97; filled dots, left axis) and number of templates (open '
          'squares, right axis). Bottom: the distribution of the fitting factor '
          'over the 10,000 injections (higher is better; white tick: median; red dotted line: 0.97). The axis stops at '
          '0.96, below which lie 0.4 % of isobank\'s injections and 2.4-3.1 % of each mbank bank\'s. isobank: the '
          'paper\'s bank (2.27M). The 14 mbank banks are almost identical whatever was changed (median 0.989). '
          'isobank\'s median is lower (0.985), because it has 23 % fewer templates, but its tail is far thinner: '
          'its templates are spread more evenly, so fewer signals fall in gaps between them.', small)]),
    table([['what was changed', 'variants (paper value)', 'coverage', 'templates'],
           ['nothing (three repeats)', '—', '93.5, 93.7, 94.8 %', '2.84–3.09M'],
           ['loss', 'weighted loss, exponent 0.5 or 1 (paper: unweighted)', '94.5, 94.4 %', '2.99, 3.03M'],
           ['start of the constant C', '0.5 or 0.99 quantile (0.9)', '94.0, 93.5 %', '2.92, 3.01M'],
           ['margin around the box', '0.001 or 0.05 (0.01)', '93.8, 94.3 %', '3.05, 3.07M'],
           ['learning rate', '5e-4 (5e-3)', '93.4 %', '2.84M'],
           ['batch size', '1,000 (5,000)', '93.6 %', '2.98M'],
           ['training points', '50,000 (15,000)', '94.3 %', '3.04M'],
           ['placement', 'importance sampling, weights clipped at 10 or not', '94.2, 94.3 %', '2.99, 3.11M'],
           ['isobank, paper bank', '', '96.64 +/- 0.18 %', '2.27M']],
          [3.9 * cm, 6.4 * cm, 3.4 * cm, 2.8 * cm], bold_last=True),
    Spacer(1, 4),
    P('All variants fall inside, or just next to, the scatter of the paper\'s settings. The weighted loss gives the '
      'best fit of the flow and a coverage at the top of the range (94.5 %), but one repeat with no change at all '
      'did better (94.8 %), so the data cannot tell it apart from the scatter. To claim it is better we would need '
      'repeats of it.'),
    P('One earlier explanation needs correcting. Only about 4 % of mbank\'s flow samples fall inside the parameter '
      'box, and we had blamed the margin around the box. The sweep shows the margin does not matter; what changes '
      'it is the starting value of the constant C (19 % inside at the 0.5 quantile, 2 % at 0.99). It affects the '
      'volume mbank reports and how many proposals it wastes, not the bank: the volume inside the box is the same '
      'in every variant.'),

    KeepTogether([
        P('4b. Minimal-match sweep (reviewer 6RHG)', h3),
        P('Here mbank\'s flow is the paper\'s, unchanged, and only its placement is varied. Two settings control the '
          'placement:'),
        bullet(f'{B("Minimal match")} (<font name="Symbol">m</font>; paper 0.97): the match each template must guarantee to its neighbours. '
               'A higher value packs templates more closely, so the bank is larger and covers more.'),
        bullet(f'{B("Covering fraction")} (cf; paper 0.9): placement stops once this fraction of random points in '
               'the space is matched by some template at the minimal match. A higher value fills more of the gaps.'),
        figure(f'{S}/rep_mmsweep.png', 14 * cm),
        P('Figure 3. Coverage against bank size for mbank (circles: all injections, left axis; squares: chirp mass '
          'below 1.4 solar masses, right axis), for minimal matches <font name="Symbol">m</font> from 0.97 to 0.98775 and covering fractions 0.6, '
          '0.9 and 0.99; grey is the paper\'s covering fraction. Stars: isobank\'s paper bank (2.27M); dotted and '
          'dashed lines: its coverage, all injections and low chirp mass.', small)]),
    P('mbank reaches the paper\'s isobank coverage (96.6 %) only at a minimal match of 0.977–0.978, with 4.5–4.8M '
      'templates (96.19 % and 96.44 %), twice isobank\'s 2.27M; reaching 99 % takes 11.9M (<font name="Symbol">m</font> = 0.98775, 98.97 %) '
      'or 24.7M (cf 0.99, 99.50 %). Lowering the covering fraction to 0.6 brings mbank near isobank\'s size (2.49M) '
      'but its coverage drops to 92.49 %. For chirp mass below 1.4 solar masses, isobank covers 91.3 %; mbank '
      'needs 24.7M templates to match it (92.3 %). So tightening mbank\'s placement buys coverage only by adding templates '
      'everywhere, most of them where coverage was already fine: the gap between the two banks is in where the '
      'templates go, not in how many there are.'),

    KeepTogether([
        P('5. Compute cost', h2),
        P('Wall time of each step at Nikhef, from the HTCondor job history (eccentric 5D). Scoring the banks against '
          'the injections is listed apart: it is the same test for every method and not part of building a bank.'),
        figure(f'{S}/rep_compute.png', 12.5 * cm),
        P('Figure 4. Wall time x CPUs of each step, log scale. Open markers: steps; circle: construction (all '
          'steps); star: matched filtering one hour of data with the bank, in PyCBC\'s convention (table below). '
          'Banks: the paper\'s isobank bank (2.27M), mbank\'s three paper-setting runs (2.84–3.09M) and Cabass et al. '
          '(20 sub-banks, 3.41M). isobank\'s construction times are those of its retraining at Nikhef, as the paper\'s were not '
          'timed. isobank computes its training pairs inside the training job, so its '
          'preprocessing is part of training. mbank: mean of the three runs with the paper\'s settings. Cabass et al.: preprocessing is the waveforms, the SVD and the bank-size regressor; placement '
          'is counting the lattice cells. Constructing any of these banks costs less than searching a single hour '
          'of data. isobank\'s bank is the smallest, so it is the cheapest to search (929 CPU-hours, against 1,220 for '
          'mbank and 1,399 for Cabass et al.), and it covers the most.',
          small),
        table([['method', 'step', 'CPUs', 'wall time', 'CPU-hours'],
               ['isobank', 'train the flow (radius, pairs, training)', '1', '3.35 h', '3.4'],
               ['', 'place the templates', '1', '0.35 h', '0.3'],
               ['', '<b>total</b>', '', '<b>3.7 h</b>', '<b>3.7</b>'],
               ['mbank, paper settings (3 runs)', 'training set (15,000 points)', '1', '0.12 h', '0.1'],
               ['', 'train the flow', '1', '0.12–0.19 h', '0.2'],
               ['', 'place the templates', '1', '0.77–1.56 h', '0.8–1.6'],
               ['', '<b>total</b>', '', '<b>1.0–1.9 h</b>', '<b>1.0–1.9</b>'],
               ['Cabass et al., 20 sub-banks', 'waveforms, SVD, size regressor', '32', '2.99 h', '96 (requested)'],
               ['', '20 autoencoders', '1 each', '1.67 h (slowest)', '21'],
               ['', 'count the grid cells', '1 each', '0.30 h', '7'],
               ['', '<b>total</b>', '', '<b>about 5 h</b>', '<b>about 125</b>'],
               ['scoring (all methods)', '10,000 injections x 500 candidate templates', '1', '5.7 h', '5.7']],
              [4.0 * cm, 6.2 * cm, 1.4 * cm, 2.6 * cm, 2.3 * cm])]),
    Spacer(1, 4),
    P('isobank is not faster to construct than mbank. Its training costs more (3.35 h against about 10 minutes), '
      'because the metric is evaluated when the training pairs are made; its placement is two to four times faster, '
      'because it evaluates no metric at all. The difference is a few CPU-hours, once, against a search cost that '
      'grows with the number of templates for every second of data.'),
    P(B("In the paper's banks") + ', isobank has 2,267,558 templates against mbank\'s 2,953,739, and covers more '
      '(96.6 % against 94.2 %). With the cost below, searching one hour of data takes about 930 CPU-hours with '
      'isobank\'s bank and 1,210 with mbank\'s: 23 % less, for higher coverage.'),
    KeepTogether([
        P(B("Search cost, in PyCBC's convention.") + ' PyCBC filters each template against the data in the '
          'frequency domain, in blocks of 256 s sampled at 4096 Hz, with one inverse complex FFT per template and '
          'block (Usman et al. 2016, Sec. 3.3–3.4). The cost per template is therefore fixed, and the search cost '
          'is proportional to the number of templates:'),
        table([['', 'value', 'from'],
               ['samples per block, N', '2<super>20</super>', '256 s at 4096 Hz (Usman et al. 2016)'],
               ['cost per template and block', '1.0 x 10<super>8</super> flops',
                'one inverse complex FFT, 5 N log<sub>2</sub> N'],
               ['cost per template and second of data', '4.1 x 10<super>-4</super> CPU-seconds',
                '128 s of each block kept (pycbc_inspiral padding), 2 x 10<super>9</super> flops per second per core'],
               ['templates saved by isobank', '0.69M', 'the paper\'s banks: 2.27M against 2.95M, at higher coverage'],
               ['search time saved', '~280 CPU-seconds per second of data', '0.69M x 4.1 x 10<super>-4</super>'],
               ['extra cost to construct isobank\'s bank', '~8 x 10<super>3</super> CPU-seconds',
                '2.2 CPU-hours more than mbank (table above)'],
               ['<b>break-even</b>', '<b>~30 s of data</b>', '8 x 10<super>3</super> / 280']],
              [4.6 * cm, 4.4 * cm, 7.5 * cm])]),
    Spacer(1, 4),
    P('The estimate is robust: even if it is off by a factor of 100, isobank\'s extra training is repaid after under an '
      'hour of searched data, against observing runs of months. Real pipelines use further tricks (SVD of the '
      'templates, hierarchical searches), but their cost still grows with the number of templates.'),
    P('Cabass et al.\'s cost is dominated by the waveforms needed for the SVD. Its 96 CPU-hours are requested, not '
      'measured, and include the regressor we use to count bank sizes, so the method itself costs less; it is still '
      'well above the other two.', small),

    KeepTogether([
        P('6. Loss ablation and retraining variance (Alexandra)', h2),
        P('Alexandra retrained the flow with each term of the loss removed in turn, and with each coefficient '
          '(<font name="Symbol">b</font><sub>loc</sub>, <font name="Symbol">b</font><sub>bnd</sub>, <font name="Symbol">b</font><sub>edge</sub>) halved or doubled, and the tail weight <font name="Symbol">g</font> (nominally 3) set to 0, 1 and 10. '
          'Every setting was trained at least twice. Each flow was placed so that its bank has 2.27M templates, the '
          'paper\'s size, and scored on 4,000 injections, so the settings are compared at equal cost.'),
        figure(f'{S}/rep_ablation.png', 13 * cm),
        P('Figure 5. Missed injections (FF below 0.97) for each setting: mean and standard deviation over its '
          'retrainings (<font name="Symbol">s</font>, which includes each run\'s injection error of about 0.28 points). Blue band: the full objective, five retrainings, 3.36 \u00b1 0.38 %. Orange: one term '
          'removed; green: one coefficient changed. Numbers from Alexandra\'s tables.', small)]),
    bullet(f'{B("Every local term matters.")} Without L<sub>loc</sub>, L<sub>bnd</sub> or L<sub>edge</sub>, '
           'the bank misses 1.1 to 1.8 points more; with L<sub>glob</sub> alone, 3.5 points more. '
           'Removing L<sub>glob</sub> makes no measurable difference.'),
    bullet(f'{B("The coefficients are robust")} within a factor of two, except that halving '
           '<font name="Symbol">b</font><sub>edge</sub> hurts (+0.9 points). Without tail weighting (<font name="Symbol">g</font> = 0) the bank misses 0.9 points more, and with '
           '<font name="Symbol">g</font> = 10, 1.9 points more; <font name="Symbol">g</font> = 1 makes no difference.'),
    bullet(f'{B("The radius has to be set for each trained flow.")} At the same radius, retrained flows give banks '
           'of very different sizes (1.6–5.6M at 1.45 r<sub>target</sub>), because each has a slightly different '
           'latent scale and the number of templates grows as r<super>-5</super> in five dimensions. At the same '
           'bank size their coverage is stable (miss rate 2.75–3.73 %). On the aligned-spin space the radius is '
           'stable (0.91–1.05 r<sub>target</sub>).'),
    P('Alexandra\'s miss rates compare each injection with its 200 nearest templates only. The miss rate still falls '
      'between 100 and 200 nearest (from 4.4 % to 3.4 % for the full objective), so they are upper bounds; the '
      'comparison between settings is fair, as every setting is scored the same way.', small),
    bullet(f'{B("Our Nikhef retrainings agree.")} Three new network seeds at r<sub>target</sub> gave 2.74M, 6.11M '
           'and 6.34M templates (reference 3.01M), with coverage 97.5–98.9 %; re-placing the reference flow with '
           'two other placement seeds changed its size by under 0.6 % and its coverage by 0.03 points. The variance '
           'is in the training, not the placement. One run (6.11M, 97.5 %) lies above Alexandra\'s curve of miss '
           'rate against size.'),
    P('Note: these banks were placed with the best-validation checkpoint (flow_best.pt), which Alexandra found inflates '
      'the latent distances in some runs; her runs use the final epoch.', small),

    P('7. Points to keep in mind', h2),
    bullet(f'{B("The eccentric space suits SVD methods.")} TaylorF2Ecc is inspiral only, so every waveform has the '
           'same amplitude shape; the aligned-spin space shows what happens when it does not (section 2).'),
    bullet(f'{B("Bank sizes.")} isobank\'s numbers are the paper\'s bank (2.27M templates) throughout. '
           'Cabass et al. bank sizes are counted grid cells (converged to about 2 %), not banks built in full.'),
    bullet(f'{B("One run per mbank variant.")} Differences below about 1 percentage point are within the scatter.'),

    P('Where everything is', h2),
    P(f'Code and instructions on the iclr branch: {code("autoencoder_comparison/svd_ae/")} (Cabass et al.), '
      f'{code("svd_baseline/")} (Alexandra\'s SVD lattice), {code("mbank_comparison/tuning/")} (mbank sweep). '
      f'Figures: {code("autoencoder_comparison/runs/eccentric_5d/svd_ae/ff_cdf_svd_vs_isobank.pdf")} and '
      f'{code("mbank_comparison/tuning/results/tuning_summary.pdf")}.', small),
]
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title='Baselines for the reviewer', author='').build(story)
print(OUT)
