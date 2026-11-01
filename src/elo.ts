import { TIERS } from './encode.ts'

/** Elo estimate from a score against an opponent of rating R, clamped away from 0/1. */
export const elo = (score: number, R: number) => {
  const s = Math.min(0.99, Math.max(0.01, score))
  return R + 400 * Math.log10(s / (1 - s))
}

export type EvalEntry = { tier: string; score: number }
const WINDOW = 40, THRESHOLD = 0.55

/** Next tier iff the last 40 eval games are all vs `current` and average ≥ 0.55; never demotes. */
export function promote(history: EvalEntry[], current: string): string {
  const last = history.slice(-WINDOW)
  if (last.length < WINDOW || last.some(e => e.tier !== current)) return current
  const mean = last.reduce((s, e) => s + e.score, 0) / WINDOW
  const i = TIERS.findIndex(t => t.name === current)
  return mean >= THRESHOLD && i + 1 < TIERS.length ? TIERS[i + 1].name : current
}

/** Mean score of the last 40 eval games against `tier` only; 0.5 when there are none yet. */
export function rollingScore(history: EvalEntry[], tier: string): number {
  const mine = history.filter(e => e.tier === tier).slice(-WINDOW)
  return mine.length ? mine.reduce((s, e) => s + e.score, 0) / mine.length : 0.5
}
