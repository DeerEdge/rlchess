import { TIERS } from './encode.ts'

/** Standard Elo update after one game scored `s` against an opponent rated R. */
export const update = (elo: number, R: number, s: number, K = 32) => elo + K * (s - 1 / (1 + 10 ** ((R - elo) / 400)))

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
