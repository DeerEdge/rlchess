import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { TIERS } from './encode.ts'
import { createNet, loadWeights, saveWeights } from './net.ts'
import { agentPlayer, playGame, type Player, type Result } from './game.ts'
import { ReplayBuffer } from './buffer.ts'
import { playerFor, EngineTimeout } from './uci.ts'
import { trainEpoch } from './train.ts'
import { update, promote } from './elo.ts'
import { publish, restore, liveReporter, type GameRecord, type Record3 } from './store.ts'
import { Chess } from 'chess.js'
import type { EvalEntry } from './elo.ts'

const env = (k: string, d: number) => +(process.env[k] ?? d)
const GAMES = env('RLCHESS_GAMES', 8), SIMS = env('RLCHESS_SIMS', 40), EVAL_GAMES = env('RLCHESS_EVAL', 10)
const BUDGET_MS = env('RLCHESS_BUDGET_MS', 5 * 60e3) // stop starting training games after this; eval always runs
const LIVE_MS = env('RLCHESS_LIVE_MS', 180e3) // live-key cadence; KV writes are metered (free plan: 1000/day)
const TRAIN_SAMPLES = 4096, MAX_PLIES = 200, ENGINE = { movetime: 100, depth: 8 }, EVAL_TEMPERATURE_PLIES = 4
const ANNOTATOR = { name: 'stockfish', rating: 0, cmd: ['stockfish'] }, ANNOTATE_PLIES = 100 // the page replays at most 100 plies
const FILES = { state: 'state.json', weights: 'weights.json', buffer: 'buffer.jsonl' }

type State = { run: number; tier: string; elo: number; evalHistory: EvalEntry[]; record: Record<string, Record3>; gamesTotal: number }
type Game = Awaited<ReturnType<typeof playGame>> & { agentWhite: boolean; against?: string }

/** Play up to n games; a game whose engine goes silent is dropped, and no new game starts after `deadline`. */
export async function playMany(n: number, factory: (i: number) => Promise<Game>, deadline = Infinity): Promise<Game[]> {
  const out: Game[] = []
  for (let i = 0; i < n && Date.now() < deadline; i++) {
    try { out.push(await factory(i)) } catch (e) { if (e instanceof EngineTimeout) console.warn(`dropped game ${i}: ${e.message}`); else throw e }
  }
  return out
}

export const score = (g: { result: Result; agentWhite: boolean }) => g.result === '1/2-1/2' ? 0.5 : (g.result === '1-0') === g.agentWhite ? 1 : 0

/** Training opponent: the current tier or one of its neighbours, so the net sees a spread of strengths. */
export function trainingTier(current: string) {
  const i = TIERS.findIndex(t => t.name === current)
  const choices = [i - 1, i, i + 1].filter(j => j >= 0 && j < TIERS.length)
  return TIERS[choices[Math.floor(Math.random() * choices.length)]]
}

/** Centipawn eval (white's view) of the start position and each ply of a game, for the page's eval bar. */
export async function annotate(games: GameRecord[], evaluate: (fen: string) => Promise<number>) {
  for (const g of games) {
    const c = new Chess(); c.loadPgn(g.pgn)
    const moves = c.history().slice(0, ANNOTATE_PLIES), r = new Chess(), evals = [await evaluate(r.fen())]
    for (const m of moves) { r.move(m); evals.push(await evaluate(r.fen())) }
    g.evals = evals
  }
}

/** Score / win / draw rates over the last 40 eval games vs `tier`; neutral when there are none. */
export function rates(history: EvalEntry[], tier: string) {
  const last = history.filter(e => e.tier === tier).slice(-40), n = last.length
  const count = (v: number) => last.filter(e => e.score === v).length / n
  return n ? { score40: last.reduce((s, e) => s + e.score, 0) / n, win40: count(1), draw40: count(0.5) } : { score40: 0.5, win40: 0, draw40: 0 }
}

async function main() {
  const t0 = Date.now()
  if (!existsSync(FILES.state) && await restore({ 'worker-state': FILES.state, weights: FILES.weights, buffer: FILES.buffer })) console.log('restored state from kv')
  const state: State = existsSync(FILES.state) ? JSON.parse(readFileSync(FILES.state, 'utf8')) : { run: 0, tier: TIERS[0].name, elo: TIERS[0].rating, evalHistory: [], record: {}, gamesTotal: 0 }
  state.elo ??= TIERS[0].rating // states written before per-game Elo
  const tier = TIERS.find(t => t.name === state.tier)!, ts = new Date().toISOString()
  const net = createNet(); if (existsSync(FILES.weights)) await loadWeights(net, FILES.weights)
  const buffer = ReplayBuffer.load(FILES.buffer)
  const explore = agentPlayer(net, SIMS, { noise: true }), exploit = agentPlayer(net, SIMS, { temperaturePlies: EVAL_TEMPERATURE_PLIES })
  const run = state.run + 1, live = liveReporter(undefined, LIVE_MS)
  const watch = (phase: 'play' | 'eval', game: number, games: number, against: string, agentWhite: boolean) =>
    (c: Chess) => live({ phase, run, tier: tier.name, game, games, against, agentWhite, pgn: c.pgn() })
  // one engine process per game: a timed-out engine is killed, so it must never be shared across games
  const vs = async (agent: Player, i: number, against = tier, phase: 'play' | 'eval' = 'play'): Promise<Game> => {
    const opp = await playerFor(against, ENGINE), agentWhite = i % 2 === 0, n = phase === 'eval' ? EVAL_GAMES : GAMES
    try { return { ...await playGame(agentWhite ? agent : opp, agentWhite ? opp : agent, { maxPlies: MAX_PLIES, record: true, onMove: watch(phase, i + 1, n, against.name, agentWhite) }), agentWhite, against: against.name } } finally { opp.quit() }
  }
  const games: GameRecord[] = []
  // rated eval games go first so the page's newest games are the ones that moved the rating
  const log = (kind: GameRecord['kind'], g: Game, rating: Pick<GameRecord, 'elo' | 'delta'> = {}) => games[kind === 'eval' ? 'unshift' : 'push']({ ts, tier: g.against ?? tier.name, kind, result: g.result, pgn: g.pgn, agentWhite: g.agentWhite, plies: g.plies, ...rating })

  const deadline = t0 + BUDGET_MS
  const tierGames = await playMany(Math.ceil(GAMES / 2), i => vs(explore, i, trainingTier(tier.name)), deadline)
  const selfGames = await playMany(Math.floor(GAMES / 2), async i => ({ ...await playGame(explore, explore, { maxPlies: MAX_PLIES, record: true, onMove: watch('play', tierGames.length + i + 1, GAMES, 'agent', true) }), agentWhite: true }), deadline)
  tierGames.forEach(g => log('tier', g)); selfGames.forEach(g => log('selfplay', g))
  buffer.append([...tierGames, ...selfGames].flatMap(g => g.samples))
  const subset = [...buffer.samples].sort(() => Math.random() - 0.5).slice(0, TRAIN_SAMPLES) // ponytail: good-enough shuffle
  const { loss } = await trainEpoch(net, subset, { batch: 256, lr: 1e-3, onBatch: (batch, batches, l) => live({ phase: 'train', run, tier: tier.name, batch, batches, loss: +l.toFixed(4) }) })

  const evalGames = await playMany(EVAL_GAMES, i => vs(exploit, i, tier, 'eval'))
  const rec = (state.record[tier.name] ??= { w: 0, d: 0, l: 0 })
  for (const g of evalGames) {
    const s = score(g), before = state.elo
    state.elo = update(state.elo, tier.rating, s); state.evalHistory.push({ tier: tier.name, score: s }); rec[s === 1 ? 'w' : s === 0 ? 'l' : 'd']++
    log('eval', g, { elo: Math.round(state.elo), delta: Math.round(state.elo - before) })
  }

  try { const sf = await playerFor(ANNOTATOR, ENGINE); try { await annotate(games, sf.evaluate) } finally { sf.quit() } } catch (e) { console.warn(`annotate: ${(e as Error).message}`) }

  const { score40, win40, draw40 } = rates(state.evalHistory, tier.name)
  const summary = { ts, run: ++state.run, tier: tier.name, elo: Math.round(state.elo), score40: +score40.toFixed(3), win40: +win40.toFixed(3), draw40: +draw40.toFixed(3),
    plies: Math.round(games.reduce((s, g) => s + g.plies, 0) / Math.max(1, games.length)), buffer: buffer.samples.length, seconds: Math.round((Date.now() - t0) / 1000),
    gamesTotal: (state.gamesTotal += games.length), loss: +loss.toFixed(4), record: state.record }
  state.tier = promote(state.evalHistory, state.tier)
  state.evalHistory = state.evalHistory.slice(-40)
  writeFileSync(FILES.state, JSON.stringify(state, null, 1)); await saveWeights(net, FILES.weights); buffer.save(FILES.buffer)
  await publish(summary, games, { weights: readFileSync(FILES.weights, 'utf8'), buffer: readFileSync(FILES.buffer, 'utf8'), state: readFileSync(FILES.state, 'utf8') })
  console.log(`run ${summary.run} tier ${tier.name}→${state.tier} score40 ${summary.score40} elo ${summary.elo} loss ${summary.loss} games ${games.length} buffer ${buffer.samples.length} ${Math.round((Date.now() - t0) / 1000)}s`)
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main().catch(e => { console.error(e); process.exit(1) })
