// widgets/linear-signal-data.js — pure data preparation for the initialisation widgets (histogram and signal plot).
// Nothing here draws, and nothing here draws random numbers: the weights' statistics and the six-layer activation table
// are PRECOMPUTED by the real TorC run (site/data/linear.json: `init_stats`, `signal`). This file only reshapes them.

// ---- numbers -------------------------------------------------------------------------------------
export function fmtSci(x) {
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e-3 && a < 1e5) return String(Number(x.toPrecision(4)));
  return x.toExponential(2).replace('e+', 'e');
}

// Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). Used only to draw the EXPECTED bell curve.
export function normalCdf(z) {
  const t = 1 / (1 + 0.3275911 * Math.abs(z / Math.SQRT2));
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

// ---- histogram -------------------------------------------------------------------------------------
// h = init_stats[i].histogram {lo, hi, bins, counts[], under, over}; expected counts for N(0, sigma) with `numel` draws.
export function histBins(h, numel, sigma) {
  const w = (h.hi - h.lo) / h.bins;
  return h.counts.map((count, i) => {
    const lo = h.lo + i * w, hi = lo + w;
    return { lo, hi, mid: lo + w / 2, count, expected: numel * (normalCdf(hi / sigma) - normalCdf(lo / sigma)) };
  });
}

// A short, plain label for one init_stats record.
export function statsLabel(s) {
  const base = `Linear(${s.in}, ${s.out})`;
  if (s.kaiming_default) return `${base}, default init (${s.numel} weights)`;
  return `${base}, init_std ${Number.isInteger(s.init_std) ? s.init_std.toFixed(1) : s.init_std} (${s.numel} weights)`;
}

// ---- signal through six layers -----------------------------------------------------------------------
export const SIGNAL_LABELS = { kaiming: 'Kaiming (default)', 'std_0.01': 'init_std 0.01', 'std_1.0': 'init_std 1.0' };
export const signalLabel = (rec) => SIGNAL_LABELS[rec.name] ?? (rec.kaiming_default ? SIGNAL_LABELS.kaiming : `init_std ${rec.init_std}`);

// One series per init: points (layer 0 = the input itself, 1..6 = after Linear(256,256)+ReLU) and the min/max/mean band over other seeds.
export function signalSeries(signal) {
  return signal.map((rec) => ({
    name: rec.name, label: signalLabel(rec), init_std: rec.init_std, kaiming: !!rec.kaiming_default,
    points: rec.layers.map((l) => ({ layer: l.layer, std: l.post_std, fracZero: l.frac_zero })),
    band: (rec.across_seeds ?? []).map((a) => ({ layer: a.layer, min: a.post_std_min, max: a.post_std_max, mean: a.post_std_mean })),
  }));
}

// Layer-to-layer growth factors of the primary run (layer l+1 / layer l, for l >= 1 so the input is left out).
export function growthRatios(series) {
  const p = series.points.filter((q) => q.layer >= 1);
  return p.slice(1).map((q, i) => q.std / p[i].std);
}

// Rows for the "show as table" view: layer, then primary run and across-seed range for each init.
export function signalTable(series) {
  const columns = ['layer'];
  for (const s of series) { columns.push(`${s.label}: this run`); if (s.band.length) columns.push(`${s.label}: 10 other seeds, min – max`); }
  const rows = series[0].points.map((p, idx) => {
    const r = [p.layer === 0 ? '0 (input)' : String(p.layer)];
    for (const s of series) {
      r.push(fmtSci(s.points[idx].std));
      if (s.band.length) { const b = s.band.find((x) => x.layer === p.layer); r.push(b ? `${fmtSci(b.min)} – ${fmtSci(b.max)}` : '–'); }
    }
    return r;
  });
  return { columns, rows };
}

// log10 axis helpers
export const log10 = (x) => Math.log10(x);
export function decadeTicks(lo, hi, step = 2) {
  const out = [];
  for (let e = Math.ceil(lo); e <= hi; e++) if ((e - Math.ceil(lo)) % step === 0) out.push(e);
  return out;
}
export const expLabel = (e) => (e === 0 ? '1' : `1e${e}`);
