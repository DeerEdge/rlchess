import { existsSync, readFileSync, writeFileSync, readdirSync, unlinkSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { TIERS } from './encode.ts'
import { createNet, loadWeights, saveWeights } from './net.ts'
import { agentPlayer, playGame, type Player, type Result, type Sample } from './game.ts'
import { ReplayBuffer, shuffle } from './buffer.ts'
import { playerFor, EngineTimeout } from './uci.ts'
import { trainEpoch } from './train.ts'
import { update, promote } from './elo.ts'
import { publish, restore, type GameRecord, type Record3 } from './store.ts'
import { Chess } from 'chess.js'
import type { EvalEntry } from './elo.ts'

const env = (k: string, d: number) => +(process.env[k] ?? d)
const GAMES = env('RLCHESS_GAMES', 32), SIMS = env('RLCHESS_SIMS', 40), EVAL_GAMES = env('RLCHESS_EVAL', 24), SHARDS = 4 // one process per runner core
const BUDGET_MS = env('RLCHESS_BUDGET_MS', 3.5 * 60e3) // per play cycle: stop starting games after this
const TRAIN_SAMPLES = 32768, MAX_PLIES = 200, ENGINE = { movetime: 100, depth: 8 }, EVAL_TEMPERATURE_PLIES = 4
const ANNOTATOR = { name: 'stockfish', rating: 0, cmd: ['stockfish'] }, ANNOTATE_PLIES = 100 // per-ply evals of rated games: the page's eval bar (≤100 plies) and acpl
const TEACHERS = ['stockfish-3', 'stockfish-8', 'stockfish-14', 'stockfish-20'] // always in the training pool: engine moves are the expert policy targets (Stockfish only: Sunfish/Lozza time out under 4-shard load)
const FILES = { state: 'state.json', weights: 'weights.json', buffer: 'buffer.jsonl' }

type State = { run: number; tier: string; elo: number; evalHistory: EvalEntry[]; record: Record<string, Record3>; gamesTotal: number; loss: number }
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

/** Training opponent: the rung below, the current rung, the rung above, or one of the teacher engines. */
export function trainingTier(current: string) {
  const i = TIERS.findIndex(t => t.name === current)
  const pool = [...new Set([...[i - 1, i, i + 1].filter(j => j >= 0 && j < TIERS.length).map(j => TIERS[j].name), ...TEACHERS])]
  const name = pool[Math.floor(Math.random() * pool.length)]
  return TIERS.find(t => t.name === name)!
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

/** Value target per sample: Stockfish's eval squashed to [−1, 1], from the side to move (what the value head predicts). */
export async function annotateSamples(samples: Sample[], evaluate: (fen: string) => Promise<number>) {
  for (const s of samples) { const cp = await evaluate(s.fen); s.v = +Math.tanh((s.fen.split(' ')[1] === 'w' ? cp : -cp) / 400).toFixed(3) }
}

/** Average centipawn loss of the agent's own moves over the annotated plies, each clamped to [0, 1000]: lower is stronger. */
export function acpl(games: Pick<GameRecord, 'evals' | 'agentWhite'>[]) {
  const losses = games.flatMap(g => (g.evals ?? []).slice(1).flatMap((after, p) => // ply p takes evals[p] → evals[p+1]; white moves on even plies
    (p % 2 === 0) === g.agentWhite ? [Math.max(0, Math.min(1000, (p % 2 === 0 ? 1 : -1) * (g.evals![p] - after)))] : []))
  return losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : 0
}

/** Score / win / draw rates over the last 40 eval games vs `tier`; neutral when there are none. */
export function rates(history: EvalEntry[], tier: string) {
  const last = history.filter(e => e.tier === tier).slice(-40), n = last.length
  const count = (v: number) => last.filter(e => e.score === v).length / n
  return n ? { score40: last.reduce((s, e) => s + e.score, 0) / n, win40: count(1), draw40: count(0.5) } : { score40: 0.5, win40: 0, draw40: 0 }
}

const read = <T>(file: string, fallback: T): T => existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : fallback
/** Shard files `<prefix>-<n>.json` in shard order; each is read once and deleted. */
const take = <T>(prefix: string): T[] => readdirSync('.').filter(f => new RegExp(`^${prefix}-\\d+\\.json$`).test(f))
  .sort((a, b) => parseInt(a.slice(prefix.length + 1)) - parseInt(b.slice(prefix.length + 1)))
  .map(f => { const v = read<T>(f, null as T); unlinkSync(f); return v })

type Mode = 'restore' | 'play' | 'train' | 'eval' | 'publish'
/** One run (see .github/workflows/loop.yml): restore → 3 × (4 × play ∥ → train) → 4 × eval ∥ → publish.
 *  restore: pull state from KV on a cache miss.  play <n>: training games vs the pool and self-play → play-<n>.json (samples with engine values).
 *  train: merge the play shards into the buffer, one Adam pass.  eval <n>: rated games n, n+4, … vs the current rung with the frozen protocol → eval-<n>.json.
 *  publish: rate the eval games in order, promote, publish. */
async function main(mode: Mode, shard: number) {
  const t0 = process.env.START ? +process.env.START * 1000 : Date.now(), ts = new Date(t0).toISOString() // the run's start (the workflow exports START)
  if (!existsSync(FILES.state) && await restore({ 'worker-state': FILES.state, weights: FILES.weights, buffer: FILES.buffer })) console.log('restored state from kv')
  if (mode === 'restore') return
  const state = read<State>(FILES.state, { run: 0, tier: TIERS[0].name, elo: TIERS[0].rating, evalHistory: [], record: {}, gamesTotal: 0, loss: 0 })
  const tier = TIERS.find(t => t.name === state.tier)!
  const net = createNet()
  if (mode !== 'publish' && existsSync(FILES.weights)) try { await loadWeights(net, FILES.weights) } catch { console.warn('weights do not fit this architecture: starting fresh') }
  const withStockfish = async (f: (evaluate: (fen: string) => Promise<number>) => Promise<void>) => {
    try { const sf = await playerFor(ANNOTATOR, ENGINE); try { await f(sf.evaluate) } finally { sf.quit() } } catch (e) { console.warn(`annotate: ${(e as Error).message}`) }
  }
  // one engine process per game: a timed-out engine is killed, so it must never be shared across games
  const vs = async (agent: Player, n: number, against = tier): Promise<Game> => {
    const opp = await playerFor(against, ENGINE), agentWhite = n % 2 === 0
    try { return { ...await playGame(agentWhite ? agent : opp, agentWhite ? opp : agent, { maxPlies: MAX_PLIES, record: true }), agentWhite, against: against.name } } finally { opp.quit() }
  }
  const secs = () => Math.round((Date.now() - t0) / 1000)

  if (mode === 'play') {
    const explore = agentPlayer(net, SIMS, { noise: true }), deadline = Date.now() + BUDGET_MS
    const pool = await playMany(Math.ceil(GAMES / 2), i => vs(explore, i, trainingTier(tier.name)), deadline)
    const self = await playMany(Math.floor(GAMES / 2), async () => ({ ...await playGame(explore, explore, { maxPlies: MAX_PLIES, record: true }), agentWhite: true }), deadline)
    const samples = [...pool, ...self].flatMap(g => g.samples)
    await withStockfish(ev => annotateSamples(samples, ev))
    writeFileSync(`play-${shard}.json`, JSON.stringify({ games: pool.length + self.length, samples }))
    return console.log(`play ${shard}: ${pool.length} pool + ${self.length} self-play games, ${samples.length} samples, ${secs()}s`)
  }

  if (mode === 'train') {
    const buffer = ReplayBuffer.load(FILES.buffer), shards = take<{ games: number; samples: Sample[] }>('play')
    for (const s of shards) { buffer.append(s.samples); state.gamesTotal += s.games }
    state.loss = +(await trainEpoch(net, shuffle([...buffer.samples]).slice(0, TRAIN_SAMPLES), { batch: 256, lr: 1e-3 })).loss.toFixed(4)
    writeFileSync(FILES.state, JSON.stringify(state, null, 1)); await saveWeights(net, FILES.weights); buffer.save(FILES.buffer)
    return console.log(`train: ${shards.length} shards, loss ${state.loss}, buffer ${buffer.samples.length}, ${secs()}s`)
  }

  if (mode === 'eval') {
    const exploit = agentPlayer(net, SIMS, { temperaturePlies: EVAL_TEMPERATURE_PLIES })
    const numbers = Array.from({ length: Math.max(0, Math.ceil((EVAL_GAMES - shard) / SHARDS)) }, (_, k) => shard + k * SHARDS) // colour alternates by game number
    const games: GameRecord[] = (await playMany(numbers.length, k => vs(exploit, numbers[k])))
      .map(g => ({ ts, tier: tier.name, kind: 'eval', result: g.result, pgn: g.pgn, agentWhite: g.agentWhite, plies: g.plies }))
    await withStockfish(ev => annotate(games, ev))
    writeFileSync(`eval-${shard}.json`, JSON.stringify(games))
    return console.log(`eval ${shard}: ${games.length} rated games vs ${tier.name}, ${secs()}s`)
  }

  const games = take<GameRecord[]>('eval').flat(), rec = (state.record[tier.name] ??= { w: 0, d: 0, l: 0 })
  for (const g of games) {
    const s = score(g as GameRecord & { result: Result }), before = state.elo
    state.elo = update(state.elo, tier.rating, s); state.evalHistory.push({ tier: tier.name, score: s }); rec[s === 1 ? 'w' : s === 0 ? 'l' : 'd']++
    g.elo = Math.round(state.elo); g.delta = Math.round(state.elo - before)
  }
  games.reverse() // newest first
  const buffer = readFileSync(FILES.buffer, 'utf8'), { score40, win40, draw40 } = rates(state.evalHistory, tier.name)
  const summary = { ts, run: ++state.run, tier: tier.name, elo: Math.round(state.elo), score40: +score40.toFixed(3), win40: +win40.toFixed(3), draw40: +draw40.toFixed(3),
    acpl: Math.round(acpl(games)), plies: Math.round(games.reduce((s, g) => s + g.plies, 0) / Math.max(1, games.length)), buffer: buffer.split('\n').filter(Boolean).length,
    seconds: secs(), gamesTotal: (state.gamesTotal += games.length), loss: state.loss, record: state.record }
  state.tier = promote(state.evalHistory, state.tier)
  state.evalHistory = state.evalHistory.slice(-40)
  writeFileSync(FILES.state, JSON.stringify(state, null, 1))
  await publish(summary, games, { weights: readFileSync(FILES.weights, 'utf8'), buffer, state: readFileSync(FILES.state, 'utf8') })
  console.log(`run ${summary.run} ${tier.name}→${state.tier} elo ${summary.elo} score40 ${summary.score40} acpl ${summary.acpl} rated ${games.length} loss ${summary.loss} ${summary.seconds}s`)
}

const MODES: Mode[] = ['restore', 'play', 'train', 'eval', 'publish']
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const mode = MODES.find(m => m === process.argv[2])
  if (!mode) { console.error(`usage: worker.ts ${MODES.join('|')} [shard]`); process.exit(2) }
  main(mode, +(process.argv[3] ?? 0)).catch(e => { console.error(e); process.exit(1) })
}
