import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { TIERS } from './encode.ts'
import { createNet, loadWeights, saveWeights } from './net.ts'
import { agentPlayer, playGame, type Player, type Result } from './game.ts'
import { Buffer } from './buffer.ts'
import { playerFor, EngineTimeout } from './uci.ts'
import { trainEpoch } from './train.ts'
import { elo, promote, type EvalEntry } from './elo.ts'
import { publish, type GameRecord, type Record3 } from './store.ts'

const env = (k: string, d: number) => +(process.env[k] ?? d)
const GAMES = env('RLCHESS_GAMES', 8), SIMS = env('RLCHESS_SIMS', 40), EVAL_GAMES = env('RLCHESS_EVAL', 10)
const MAX_PLIES = 200, ENGINE = { movetime: 100, depth: 8 }
const FILES = { state: 'state.json', weights: 'weights.json', buffer: 'buffer.jsonl' }

type State = { run: number; tier: string; evalHistory: EvalEntry[]; record: Record<string, Record3>; gamesTotal: number }
type Game = Awaited<ReturnType<typeof playGame>>

/** Play n games; a game whose engine goes silent is dropped, not fatal. */
export async function playMany(n: number, factory: (i: number) => Promise<Game>): Promise<Game[]> {
  const out: Game[] = []
  for (let i = 0; i < n; i++) {
    try { out.push(await factory(i)) } catch (e) { if (e instanceof EngineTimeout) console.warn(`dropped game ${i}: ${e.message}`); else throw e }
  }
  return out
}

const scoreFor = (result: Result, agentIsWhite: boolean) => result === '1/2-1/2' ? 0.5 : (result === '1-0') === agentIsWhite ? 1 : 0

async function main() {
  const state: State = existsSync(FILES.state) ? JSON.parse(readFileSync(FILES.state, 'utf8')) : { run: 0, tier: TIERS[0].name, evalHistory: [], record: {}, gamesTotal: 0 }
  const tier = TIERS.find(t => t.name === state.tier)!, ts = new Date().toISOString()
  const net = createNet(); if (existsSync(FILES.weights)) await loadWeights(net, FILES.weights)
  const buffer = Buffer.load(FILES.buffer), opponent = await playerFor(tier, ENGINE)
  const explore = agentPlayer(net, SIMS, { noise: true }), exploit = agentPlayer(net, SIMS, { temperaturePlies: 0 })
  const vs = (agent: Player, i: number) => i % 2 === 0 ? playGame(agent, opponent, { maxPlies: MAX_PLIES, record: true }) : playGame(opponent, agent, { maxPlies: MAX_PLIES, record: true })
  const games: GameRecord[] = [], log = (kind: GameRecord['kind'], g: Game) => games.push({ ts, tier: tier.name, kind, result: g.result, pgn: g.pgn })

  const tierGames = await playMany(Math.ceil(GAMES / 2), i => vs(explore, i))
  const selfGames = await playMany(Math.floor(GAMES / 2), () => playGame(explore, explore, { maxPlies: MAX_PLIES, record: true }))
  tierGames.forEach(g => log('tier', g)); selfGames.forEach(g => log('selfplay', g))
  buffer.append([...tierGames, ...selfGames].flatMap(g => g.samples))
  const { loss } = await trainEpoch(net, buffer.samples, { batch: 256, lr: 1e-3 })

  const evalGames = await playMany(EVAL_GAMES, i => vs(exploit, i))
  const rec = (state.record[tier.name] ??= { w: 0, d: 0, l: 0 })
  evalGames.forEach((g, i) => {
    const s = scoreFor(g.result, i % 2 === 0)
    state.evalHistory.push({ tier: tier.name, score: s }); rec[s === 1 ? 'w' : s === 0 ? 'l' : 'd']++; log('eval', g)
  })
  opponent.quit()

  const last40 = state.evalHistory.slice(-40), score40 = last40.reduce((s, e) => s + e.score, 0) / Math.max(1, last40.length)
  const summary = { ts, run: ++state.run, tier: tier.name, elo: Math.round(elo(score40, tier.rating)), score40: +score40.toFixed(3), gamesTotal: (state.gamesTotal += games.length), loss: +loss.toFixed(4), record: state.record }
  state.tier = promote(state.evalHistory, state.tier)
  state.evalHistory = state.evalHistory.slice(-40)
  writeFileSync(FILES.state, JSON.stringify(state, null, 1)); await saveWeights(net, FILES.weights); buffer.save(FILES.buffer)
  await publish(summary, games, { weightsPath: FILES.weights })
  console.log(`run ${summary.run} tier ${tier.name}→${state.tier} score40 ${summary.score40} elo ${summary.elo} loss ${summary.loss} games ${games.length} buffer ${buffer.samples.length}`)
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main().catch(e => { console.error(e); process.exit(1) })
