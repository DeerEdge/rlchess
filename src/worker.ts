import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { TIERS } from './encode.ts'
import { createNet, loadWeights, saveWeights } from './net.ts'
import { agentPlayer, playGame, type Player, type Result } from './game.ts'
import { ReplayBuffer } from './buffer.ts'
import { playerFor, EngineTimeout } from './uci.ts'
import { trainEpoch } from './train.ts'
import { elo, promote, rollingScore } from './elo.ts'
import { publish, restore, type GameRecord, type Record3 } from './store.ts'
import type { EvalEntry } from './elo.ts'

const env = (k: string, d: number) => +(process.env[k] ?? d)
const GAMES = env('RLCHESS_GAMES', 8), SIMS = env('RLCHESS_SIMS', 40), EVAL_GAMES = env('RLCHESS_EVAL', 10)
const BUDGET_MS = env('RLCHESS_BUDGET_MS', 5 * 60e3) // stop starting training games after this; eval always runs
const TRAIN_SAMPLES = 4096, MAX_PLIES = 200, ENGINE = { movetime: 100, depth: 8 }, EVAL_TEMPERATURE_PLIES = 4
const FILES = { state: 'state.json', weights: 'weights.json', buffer: 'buffer.jsonl' }

type State = { run: number; tier: string; evalHistory: EvalEntry[]; record: Record<string, Record3>; gamesTotal: number }
type Game = Awaited<ReturnType<typeof playGame>> & { agentWhite: boolean }

/** Play up to n games; a game whose engine goes silent is dropped, and no new game starts after `deadline`. */
export async function playMany(n: number, factory: (i: number) => Promise<Game>, deadline = Infinity): Promise<Game[]> {
  const out: Game[] = []
  for (let i = 0; i < n && Date.now() < deadline; i++) {
    try { out.push(await factory(i)) } catch (e) { if (e instanceof EngineTimeout) console.warn(`dropped game ${i}: ${e.message}`); else throw e }
  }
  return out
}

export const score = (g: { result: Result; agentWhite: boolean }) => g.result === '1/2-1/2' ? 0.5 : (g.result === '1-0') === g.agentWhite ? 1 : 0

async function main() {
  const t0 = Date.now()
  if (!existsSync(FILES.state) && await restore({ 'worker-state': FILES.state, weights: FILES.weights, buffer: FILES.buffer })) console.log('restored state from kv')
  const state: State = existsSync(FILES.state) ? JSON.parse(readFileSync(FILES.state, 'utf8')) : { run: 0, tier: TIERS[0].name, evalHistory: [], record: {}, gamesTotal: 0 }
  const tier = TIERS.find(t => t.name === state.tier)!, ts = new Date().toISOString()
  const net = createNet(); if (existsSync(FILES.weights)) await loadWeights(net, FILES.weights)
  const buffer = ReplayBuffer.load(FILES.buffer)
  const explore = agentPlayer(net, SIMS, { noise: true }), exploit = agentPlayer(net, SIMS, { temperaturePlies: EVAL_TEMPERATURE_PLIES })
  // one engine process per game: a timed-out engine is killed, so it must never be shared across games
  const vs = async (agent: Player, i: number): Promise<Game> => {
    const opp = await playerFor(tier, ENGINE), agentWhite = i % 2 === 0
    try { return { ...await playGame(agentWhite ? agent : opp, agentWhite ? opp : agent, { maxPlies: MAX_PLIES, record: true }), agentWhite } } finally { opp.quit() }
  }
  const games: GameRecord[] = [], log = (kind: GameRecord['kind'], g: Game) => games.push({ ts, tier: tier.name, kind, result: g.result, pgn: g.pgn })

  const deadline = t0 + BUDGET_MS
  const tierGames = await playMany(Math.ceil(GAMES / 2), i => vs(explore, i), deadline)
  const selfGames = await playMany(Math.floor(GAMES / 2), async () => ({ ...await playGame(explore, explore, { maxPlies: MAX_PLIES, record: true }), agentWhite: true }), deadline)
  tierGames.forEach(g => log('tier', g)); selfGames.forEach(g => log('selfplay', g))
  buffer.append([...tierGames, ...selfGames].flatMap(g => g.samples))
  const subset = [...buffer.samples].sort(() => Math.random() - 0.5).slice(0, TRAIN_SAMPLES) // ponytail: good-enough shuffle
  const { loss } = await trainEpoch(net, subset, { batch: 256, lr: 1e-3 })

  const evalGames = await playMany(EVAL_GAMES, i => vs(exploit, i))
  const rec = (state.record[tier.name] ??= { w: 0, d: 0, l: 0 })
  for (const g of evalGames) { const s = score(g); state.evalHistory.push({ tier: tier.name, score: s }); rec[s === 1 ? 'w' : s === 0 ? 'l' : 'd']++; log('eval', g) }

  const score40 = rollingScore(state.evalHistory, tier.name)
  const summary = { ts, run: ++state.run, tier: tier.name, elo: Math.round(elo(score40, tier.rating)), score40: +score40.toFixed(3), gamesTotal: (state.gamesTotal += games.length), loss: +loss.toFixed(4), record: state.record }
  state.tier = promote(state.evalHistory, state.tier)
  state.evalHistory = state.evalHistory.slice(-40)
  writeFileSync(FILES.state, JSON.stringify(state, null, 1)); await saveWeights(net, FILES.weights); buffer.save(FILES.buffer)
  await publish(summary, games, { weights: readFileSync(FILES.weights, 'utf8'), buffer: readFileSync(FILES.buffer, 'utf8'), state: readFileSync(FILES.state, 'utf8') })
  console.log(`run ${summary.run} tier ${tier.name}→${state.tier} score40 ${summary.score40} elo ${summary.elo} loss ${summary.loss} games ${games.length} buffer ${buffer.samples.length} ${Math.round((Date.now() - t0) / 1000)}s`)
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main().catch(e => { console.error(e); process.exit(1) })
