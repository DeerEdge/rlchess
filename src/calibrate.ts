import { writeFileSync, readFileSync } from 'node:fs'
import { availableParallelism } from 'node:os'
import { TIERS, type Tier } from './encode.ts'
import { playGame } from './game.ts'
import { playerFor, EngineTimeout } from './uci.ts'
import { fit, type Match } from './elo.ts'

const GAMES = +(process.env.CALIBRATE_GAMES ?? 40), ENGINE = { movetime: 100, depth: 8 }, MAX_PLIES = 200 // the loop's frozen protocol
const ANCHOR = TIERS.find(t => t.anchor)!
// each rung vs the next two, and the anchor vs every rung, so the anchor is tied to the whole ladder however far it sits from its neighbours
const PAIRS = [...new Map([...TIERS.flatMap((t, i) => [TIERS[i + 1], TIERS[i + 2]].filter(Boolean).map(u => [t, u] as const)), ...TIERS.filter(t => t !== ANCHOR).map(t => [ANCHOR, t] as const)]
  .map(([a, b]) => [[a.name, b.name].sort().join('|'), [a, b] as const])).values()]

/** One game between two rungs, colours by game number; null if an engine went silent. Returns a's score. */
async function game(a: Tier, b: Tier, n: number): Promise<number | null> {
  const [pa, pb] = await Promise.all([playerFor(a, ENGINE), playerFor(b, ENGINE)]), aWhite = n % 2 === 0
  try {
    const { result } = await playGame(aWhite ? pa : pb, aWhite ? pb : pa, { maxPlies: MAX_PLIES, record: false })
    return result === '1/2-1/2' ? 0.5 : (result === '1-0') === aWhite ? 1 : 0
  } catch (e) { if (e instanceof EngineTimeout) return null; throw e } finally { pa.quit(); pb.quit() }
}

/** Play every pairing GAMES times on all cores, fit Bradley–Terry ratings anchored on the anchor rung, write them into tiers.json. */
async function main() {
  const jobs = PAIRS.flatMap(([a, b]) => Array.from({ length: GAMES }, (_, n) => ({ a, b, n }))), matches: Match[] = []
  let next = 0
  await Promise.all(Array.from({ length: availableParallelism() }, async () => {
    while (next < jobs.length) {
      const { a, b, n } = jobs[next++], s = await game(a, b, n)
      if (s !== null) matches.push({ a: a.name, b: b.name, score: s })
    }
  }))
  const ratings = fit(matches, [ANCHOR.name, ANCHOR.rating])
  const pairs = PAIRS.map(([a, b]) => { const ms = matches.filter(m => m.a === a.name && m.b === b.name); return { a: a.name, b: b.name, games: ms.length, score: +(ms.reduce((s, m) => s + m.score, 0) / ms.length).toFixed(3) } })
  const ladder = (JSON.parse(readFileSync('tiers.json', 'utf8')) as Tier[]).map(t => ({ ...t, rating: Math.round(ratings[t.name] / 10) * 10 })).sort((a, b) => a.rating - b.rating)
  writeFileSync('tiers.json', '[\n' + ladder.map(t => '  ' + JSON.stringify(t)).join(',\n') + '\n]\n')
  writeFileSync('calibration.json', JSON.stringify({ protocol: { ...ENGINE, maxPlies: MAX_PLIES, gamesPerPair: GAMES }, anchor: ANCHOR.name, pairs, ratings: Object.fromEntries(ladder.map(t => [t.name, t.rating])) }, null, 1) + '\n')
  console.table(pairs); console.table(ladder.map(t => ({ name: t.name, rating: t.rating })))
}

main().catch(e => { console.error(e); process.exit(1) })
