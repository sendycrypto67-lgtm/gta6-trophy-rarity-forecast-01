// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).
//
// Idea: a trophy's earn rate is earned / tracked. Both move. Players keep
// arriving (the tracked count grows and dilutes every rate), and players
// keep playing (the earned count grows as they reach each trophy). So the
// rate of a trophy is not a level, it is a race between dilution and
// progress, and the race slows as log time passes: early on a trophy can
// double in an hour, a week in it barely moves. That is why the forecast
// is fit in logit space against log time, per trophy, and not carried
// forward flat.
//
// What the program does, per trophy:
//   1. It takes the case's reads, compresses them to change points (a
//      read only counts when the tracked count or a rate moved), and fits
//      a weighted line of logit(rate) on ln(1 + hours since release),
//      over the last 48 hours, recent reads weighted more.
//   2. It shrinks each trophy's slope toward the common slope across
//      trophies (the median of the per-trophy slopes). A trophy with a
//      noisy or flat history borrows the common shape; a trophy with a
//      clean, long history keeps its own.
//   3. It reads the shrunk line at the target for the median.
//   4. It spreads the five quantiles from the fit's residual, a random
//      walk term that grows with the horizon, and a staleness term:
//      Exophase refreshes its counts in jumps, so a page that has not
//      changed in hours is likely to refresh soon, and a refresh grows
//      the tracked count and dilutes the rates. A stale page therefore
//      gets a wider band below the median than above it, and a small
//      downward shift of the median, both scaled by the jump size the
//      history shows (or a threefold prior with none seen). Where reads
//      change every few minutes this term is silent.
//   5. The platinum and the 100% trophy are handled as edges: the
//      100% trophy stays near 100%, a trophy with no earners yet gets a
//      small median from the lowest positive rate and a wide band.
//
// Everything is deterministic: the same case file gives the same bytes.
// Standard library only, no network, no randomness.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: hours are counted from here, so log time is defined. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** How far back the fit looks, in hours. */
const FIT_WINDOW_H = 48;
/** Half-life of the recency weight, in hours. */
const RECENCY_HALF_LIFE_H = 24;
/** Random-walk spread, in logit units per square-root day. */
const SIGMA_RW = 0.35;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
const expit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

interface Fit {
  slope: number;
  intercept: number;
  residSd: number;
  nEff: number;
  xBar: number;
  sxx: number;
}

/** Weighted least squares of y on x. Returns slope, intercept and residual spread. */
function wls(xs: number[], ys: number[], ws: number[]): Fit {
  const sw = ws.reduce((a, b) => a + b, 0);
  const xBar = xs.reduce((a, x, i) => a + ws[i] * x, 0) / sw;
  const yBar = ys.reduce((a, y, i) => a + ws[i] * y, 0) / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - xBar) * (xs[i] - xBar);
    sxy += ws[i] * (xs[i] - xBar) * (ys[i] - yBar);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = yBar - slope * xBar;
  let sse = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    sse += ws[i] * r * r;
  }
  const dof = Math.max(1, xs.length - 2);
  const residSd = Math.sqrt(sse / (sw * (dof / Math.max(1, xs.length))));
  const nEff = (sw * sw) / ws.reduce((a, w) => a + w * w, 0);
  return { slope, intercept, residSd, nEff, xBar, sxx };
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();

for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const atMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const targetHours = (targetMs - RELEASE_MS) / 3600000;
  const xTarget = Math.log(1 + Math.max(0, targetHours));

  // Compress the history to change points: a read counts only when the
  // tracked count or some rate moved. The launch-night page sat unchanged
  // for hours, and those identical reads carry no information a fit
  // should weight.
  const reads = [...c.history].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const kept: Read[] = [];
  for (const r of reads) {
    const prev = kept[kept.length - 1];
    if (prev === undefined) {
      kept.push(r);
      continue;
    }
    const moved =
      r.players_tracked !== prev.players_tracked ||
      r.rates.some((v, i) => v !== prev.rates[i]);
    if (moved) kept.push(r);
  }
  // Always end on the last read, even when it changed nothing.
  const last = reads[reads.length - 1];
  if (kept[kept.length - 1] !== last) kept.push(last);

  // Page staleness: how long since the page last changed at all (the
  // tracked count or any rate). Exophase refreshes its counts in jumps,
  // so a page that has sat unchanged for hours is likely to refresh
  // soon, and a refresh grows the tracked count and dilutes the rates.
  // The typical jump size comes from the jumps the history shows; with
  // none seen, the prior is a threefold jump (the launch-night refresh
  // grew the tracked count about sixfold).
  let pageStaleMs = 0;
  {
    let lastChangeIdx = 0;
    for (let j = 1; j < kept.length; j++) {
      const prev = kept[j - 1];
      const cur = kept[j];
      const moved =
        cur.players_tracked !== prev.players_tracked ||
        cur.rates.some((v, i) => v !== prev.rates[i]);
      if (moved) lastChangeIdx = j;
    }
    pageStaleMs = atMs - Date.parse(kept[lastChangeIdx].at);
  }
  const pageStaleH = pageStaleMs / 3600000;
  const jumpRatios: number[] = [];
  for (let j = 1; j < kept.length; j++) {
    const prev = kept[j - 1].players_tracked;
    const cur = kept[j].players_tracked;
    if (prev > 0 && cur / prev > 1.2) jumpRatios.push(cur / prev);
  }
  const jumpFactor = jumpRatios.length > 0 ? medianOf(jumpRatios) : 3.0;

  // Per-trophy series in the fit window, and the common slope across
  // trophies that actually moved.
  const windowStartMs = atMs - FIT_WINDOW_H * 3600000;
  const slopes: number[] = [];
  interface Series {
    xs: number[];
    ys: number[];
    ws: number[];
    lastRate: number;
    lastEarned: number;
  }
  const series: Series[] = c.trophies.map((t, i) => {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    for (const r of kept) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      const tMs = Date.parse(r.at);
      if (tMs < windowStartMs) continue;
      const hours = (tMs - RELEASE_MS) / 3600000;
      const ageH = Math.max(0, (atMs - tMs) / 3600000);
      xs.push(Math.log(1 + Math.max(0, hours)));
      ys.push(logit(rate));
      ws.push(Math.pow(0.5, ageH / RECENCY_HALF_LIFE_H));
    }
    const lastRate = last.rates[i] ?? 0;
    const lastEarned = (lastRate / 100) * last.players_tracked;
    return { xs, ys, ws, lastRate, lastEarned };
  });
  for (const s of series) {
    if (s.xs.length >= 3) {
      const f = wls(s.xs, s.ys, s.ws);
      const range = Math.max(...s.ys) - Math.min(...s.ys);
      if (range > 0.05 && Number.isFinite(f.slope)) slopes.push(f.slope);
    }
  }
  const commonSlope = medianOf(slopes);

  // The lowest positive rate in the last read, for trophies with no
  // earners yet.
  const positiveRates = series.map((s) => s.lastRate).filter((r) => r > 0.01);
  const minPositive = positiveRates.length > 0 ? Math.min(...positiveRates) : 0.1;

  const out = c.trophies.map((t, i) => {
    const s = series[i];
    let med: number;
    let sigLo: number;
    let sigHi: number;

    if (s.lastRate >= 99.9) {
      // A trophy every tracked player holds: it stays near 100%. The
      // band is tight above the median and wider below it, because a
      // flood of new players who have not played yet is the one way it
      // falls, and the logit scale punishes missing high far more than
      // missing low.
      med = 99.9;
      sigLo = 0.5;
      sigHi = 0.08;
    } else if (s.lastEarned < 0.5 && s.lastRate <= 0.01) {
      // No one has earned it yet. A small median, a wide band.
      med = Math.max(0.02, 0.3 * minPositive);
      sigLo = 1.2;
      sigHi = 1.2;
    } else {
      const f = s.xs.length >= 2 ? wls(s.xs, s.ys, s.ws) : null;
      let slope = commonSlope;
      let residSd = 0.3;
      let nEff = 1;
      let xBar = f !== null ? f.xBar : Math.log(1 + Math.max(0, (atMs - RELEASE_MS) / 3600000));
      let sxx = 0;
      if (f !== null) {
        residSd = f.residSd;
        nEff = f.nEff;
        xBar = f.xBar;
        sxx = f.sxx;
        // Shrink the trophy's own slope toward the common slope. A
        // noisy fit or a short history borrows more of the common shape.
        let w = nEff / (nEff + 6);
        if (residSd > 0.6) w *= 0.5;
        slope = w * f.slope + (1 - w) * commonSlope;
      }
      // Anchor the line at the recent level, then read it at the target.
      const yBar = s.ys.reduce((a, y, j) => a + s.ws[j] * y, 0) / Math.max(1e-9, s.ws.reduce((a, b) => a + b, 0));
      const xBarW = s.xs.reduce((a, x, j) => a + s.ws[j] * x, 0) / Math.max(1e-9, s.ws.reduce((a, b) => a + b, 0));
      const anchor = s.ys.length > 0 ? yBar - slope * xBarW : logit(s.lastRate);
      let medLogit = anchor + slope * xTarget;

      // Staleness: the page has sat unchanged for hours, so a refresh
      // is overdue. A refresh grows the tracked count by about the
      // jump factor and dilutes every rate, so the median shifts down
      // by part of that dilution (scaled by how likely a refresh is
      // inside the horizon) and the lower band widens by most of it.
      // Where reads change every few minutes this term is silent.
      const staleFrac = clamp(pageStaleH / 4, 0, 1);
      const refreshProb = clamp(c.horizon_minutes / Math.max(30, 240 - pageStaleH * 60), 0, 1);
      let staleLo = 0;
      if (pageStaleH > 2) {
        const shift = 0.5 * Math.log(jumpFactor) * refreshProb * staleFrac;
        medLogit -= shift;
        staleLo = 0.9 * Math.log(jumpFactor) * staleFrac;
      }
      const leverage = sxx > 1e-12 ? 1 + 1 / Math.max(1, nEff) + ((xTarget - xBar) * (xTarget - xBar)) / sxx : 2;
      const sigFit = Math.max(0.15, residSd * Math.sqrt(leverage));
      const sigRw = SIGMA_RW * Math.sqrt(horizonDays);
      sigLo = clamp(Math.sqrt(sigFit * sigFit + sigRw * sigRw + staleLo * staleLo), 0.25, 2.0);
      sigHi = clamp(Math.sqrt(sigFit * sigFit + sigRw * sigRw + 0.12 * staleLo * staleLo), 0.25, 2.0);
      med = expit(medLogit);
    }

    const q = Z.map((z) => {
      const s = z < 0 ? sigLo : sigHi;
      return round4(clamp(expit(logit(med) + z * s), 0.01, 99.99));
    });
    // Keep the five quantiles non-decreasing after rounding.
    for (let k = 1; k < q.length; k++) {
      if (q[k] < q[k - 1]) q[k] = q[k - 1];
    }
    return { id: t.id, q };
  });

  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: out })}\n`);
}
