"""Pure NumPy diagnostics; not training losses or replacement evaluation metrics."""
import math
import numpy as np


def ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def structure(gt, prediction, raw, *, seed=20261021, cap=4096, pair_gap=5.):
    """Same GT-driven pixel/pair sampling for every model. Predictions unclipped.

    Spearman and pair ordering describe structure, while metre-space percentile
    spans describe range after the existing whole-frame alignment. No local refit.
    """
    g, p, r = (np.asarray(x, dtype=np.float64).ravel() for x in (gt, prediction, raw))
    if not (g.size == p.size == r.size) or not all(np.isfinite(x).all() for x in (g, p, r)):
        raise ValueError('structure needs matching finite arrays')
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(g.size, min(cap, g.size), replace=False))
    gs, ps, rs = g[index], p[index], r[index]
    def corr(a, b):
        a, b = ranks(a), ranks(b)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    def span(x):
        return float(np.percentile(x, 90) - np.percentile(x, 10))
    # Random disjoint pairs (rather than quadratic all-pairs). Seed independent of model.
    perm = rng.permutation(gs.size)
    a, b = perm[:2 * (gs.size // 2):2], perm[1:2 * (gs.size // 2):2]
    dg, dp = gs[a] - gs[b], ps[a] - ps[b]
    keep = np.abs(dg) >= pair_gap
    product = dg[keep] * dp[keep]
    return dict(sample_pixels=int(gs.size), spearman=corr(gs, ps), raw_spearman=corr(gs, rs),
                gt_span_m=span(g), pred_span_m=span(p),
                span_ratio=span(p) / span(g) if span(g) > 1e-6 else None,
                pairs=int(keep.sum()),
                order_accuracy=float(np.mean(product > 0)) if product.size else None,
                order_tie_share=float(np.mean(product == 0)) if product.size else None)


def paired_summary(rows, reference, metric='mse'):
    """Per-band differences; never silently intersect unmatched IDs."""
    result = []
    for band in sorted({r['band'] for r in rows}):
        data = {}
        for row in rows:
            if row['band'] == band:
                bucket = data.setdefault(row['model'], {})
                if row['id'] in bucket:
                    raise ValueError('duplicate per-frame diagnostic row')
                bucket[row['id']] = row
        if reference not in data:
            raise ValueError(f'missing reference {reference}')
        base = data[reference]
        for model, current in data.items():
            if model == reference:
                continue
            if current.keys() != base.keys():
                raise ValueError('paired frame IDs differ')
            ids = sorted(base)
            for key in ids:
                if (current[key]['pixels'], current[key]['sequence']) != (base[key]['pixels'], base[key]['sequence']):
                    raise ValueError('paired GT populations differ')
            delta = {key: current[key][metric] - base[key][metric] for key in ids}
            total = sum(delta.values())
            positive = sum(max(x, 0.) for x in delta.values())
            def stats(selected):
                signed = sum(delta[k] for k in selected)
                excess = sum(max(delta[k], 0.) for k in selected)
                return dict(frames=len(selected), mean_mse_gap=signed / len(selected),
                            worse_frames=sum(delta[k] > 0 for k in selected),
                            signed_gap_share=signed / total if abs(total) > 1e-12 else None,
                            positive_excess_share=excess / positive if positive > 0 else None)
            ordered = sorted(ids, key=lambda k: (-delta[k], k))
            tail = ordered[:max(1, math.ceil(.1 * len(ids)))]
            sequences = sorted({r['sequence'] for r in base.values()})
            result.append(dict(model=model, reference=reference, band=band, metric=metric,
                               overall=stats(ids),
                               by_sequence={s: stats([k for k in ids if base[k]['sequence'] == s]) for s in sequences},
                               tail_ranking=f'signed {metric} difference descending, ID tie-break',
                               top10pct={**stats(tail), 'ids': tail}))
    return result


def structure_summary(rows):
    result = []
    for model, band in sorted({(r['model'], r['band']) for r in rows}):
        selected = [r for r in rows if (r['model'], r['band']) == (model, band)]
        for seq in ['ALL'] + sorted({r['sequence'] for r in selected}):
            subset = [r for r in selected if seq == 'ALL' or r['sequence'] == seq]
            out = dict(model=model, band=band, sequence=seq, frames=len(subset))
            for metric in ('spearman', 'raw_spearman', 'span_ratio', 'order_accuracy', 'order_tie_share'):
                values = [r[metric] for r in subset if r[metric] is not None]
                out[metric] = dict(defined_frames=len(values),
                                   median=float(np.median(values)) if values else None)
            result.append(out)
    return result
