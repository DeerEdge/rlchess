import { TIERS, type Tier } from './encode.ts'

/** Expected score of a player rated `a` against one rated `b`. */
export const expected = (a: number, b: number) => 1 / (1 + 10 ** ((b - a) / 400))

/** Standard Elo update after one game scored `s` against an opponent rated R. */
export const update = (elo: number, R: number, s: number, K = 32) => elo + K * (s - expected(elo, R))

/** The n rungs rated nearest to `elo`, in rating order: the rated panel, and (wider) the training pool. */
export const nearest = (elo: number, n: number, ladder: { name: string; rating: number }[] = TIERS) =>
  [...ladder].sort((a, b) => Math.abs(a.rating - elo) - Math.abs(b.rating - elo)).slice(0, n).sort((a, b) => a.rating - b.rating)

export type Match = { a: string; b: string; score: number } // a's score in one game: 1, ½ or 0

/** Maximum-likelihood Bradley–Terry (Elo) ratings from game results, `anchor` held fixed.
 *  Newton steps per player; one virtual draw per pairing keeps a sweep finite. */
export function fit(matches: Match[], anchor: [string, number], iters = 200): Record<string, number> {
  const pairings = [...new Set(matches.map(m => [m.a, m.b].sort().join('|')))].map(p => { const [a, b] = p.split('|'); return { a, b, score: 0.5 } })
  const games = [...matches, ...pairings], names = [...new Set(games.flatMap(m => [m.a, m.b]))]
  const r: Record<string, number> = Object.fromEntries(names.map(n => [n, anchor[1]]))
  for (let it = 0; it < iters; it++) {
    const g: Record<string, number> = {}, h: Record<string, number> = {}
    for (const m of games) {
      const e = expected(r[m.a], r[m.b])
      g[m.a] = (g[m.a] ?? 0) + m.score - e; g[m.b] = (g[m.b] ?? 0) - (m.score - e)
      h[m.a] = (h[m.a] ?? 0) + e * (1 - e); h[m.b] = (h[m.b] ?? 0) + e * (1 - e)
    }
    for (const n of names) if (n !== anchor[0]) r[n] += 0.5 * (400 / Math.LN10) * g[n] / h[n] // damped Newton step
  }
  return r
}

export type { Tier }
