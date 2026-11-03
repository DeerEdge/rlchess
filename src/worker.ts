import { existsSync, readFileSync, writeFileSync, readdirSync, unlinkSync } from 'node:fs'
import os from 'node:os'
import { usage } from './usage.ts'
import { benchmark } from './bench.ts'
import { fileURLToPath } from 'node:url'
import { TIERS, type Tier } from './encode.ts'
import { createNet, loadWeights, saveWeights } from './net.ts'
import { agentPlayer, playGame, type Player, type Result, type Sample } from './game.ts'
import { ReplayBuffer, shuffle } from './buffer.ts'
import { playerFor, EngineTimeout, type Analysis } from './uci.ts'
import { trainEpoch } from './train.ts'
import { update, nearest } from './elo.ts'
import { publish, restore, type GameRecord, type Record3 } from './store.ts'
import { Chess } from 'chess.js'

const env = (k: string, d: number) => +(process.env[k] ?? d)
const GAMES = env('RLCHESS_GAMES', 128), SIMS = env('RLCHESS_SIMS', 40), EVAL_GAMES = env('RLCHESS_EVAL', 24), SHARDS = 4 // one process per runner core
const PARALLEL = 8 // games in flight per process: their searches' network calls run as one batch
const BUDGET_MS = env('RLCHESS_BUDGET_MS', 3.5 * 60e3) // per play cycle: stop starting games after this
const TRAIN_SAMPLES = 60000 /* one pass over the whole buffer */, MAX_PLIES = 200, ENGINE = { movetime: 100, depth: 8 }, EVAL_TEMPERATURE_PLIES = 4
const ANNOTATOR = { name: 'stockfish', rating: 0, cmd: ['stockfish'] }, ANNOTATE_PLIES = 100 // per-ply evals of rated games: the page's eval bar (≤100 plies) and acpl
const PANEL = 4, POOL = 6, BLEND = 0.5 // rated opponents per run (one per eval shard), training opponents, share of the policy target given to Stockfish's move
const POOL_LIFT = 250 // the training pool centres this far above the agent's rating: mostly stronger opponents (the rated panel stays centred)
const FILES = { state: 'state.json', weights: 'weights.json', buffer: 'buffer.jsonl' }

type Rated = { tier: string; score: number } // one rated game: opponent and the agent's score
type State = { run: number; elo: number; evalHistory: Rated[]; record: Record<string, Record3>; gamesTotal: number; steps?: number; loss: number; policyLoss: number; valueLoss: number; benchMatch?: number; benchValue?: number } // steps: gradient steps, counted from the Lichess pretraining on
type Game = Awaited<ReturnType<typeof playGame>> & { agentWhite: boolean; against?: string }

/** Play up to n games, `parallel` at a time (their searches share batched network calls); a game whose engine goes silent is dropped,
 *  and no new game starts after `deadline`. */
export async function playMany(n: number, factory: (i: number) => Promise<Game>, deadline = Infinity, parallel = 1): Promise<Game[]> {
  const out: Game[] = []
  let next = 0
  const lane = async () => {
    while (next < n && Date.now() < deadline) {
      const i = next++
      try { out.push(await factory(i)) } catch (e) { if (e instanceof EngineTimeout) console.warn(`dropped game ${i}: ${e.message}`); else throw e }
    }
  }
  await Promise.all(Array.from({ length: parallel }, lane))
  return out
}

export const score = (g: { result: Result; agentWhite: boolean }) => g.result === '1/2-1/2' ? 0.5 : (g.result === '1-0') === g.agentWhite ? 1 : 0

/** Training opponent: one of the six rungs nearest a point POOL_LIFT above the agent. */
export function trainingPool(elo: number) {
  const pool = nearest(elo + POOL_LIFT, POOL)
  return pool[Math.floor(Math.random() * pool.length)]
}

/** Centipawn eval (white's view) of the start position and each ply of a game, for the page's eval bar. */
export async function annotate(games: GameRecord[], analyse: (fen: string) => Promise<Analysis>) {
  for (const g of games) {
    const c = new Chess(); c.loadPgn(g.pgn)
    const moves = c.history().slice(0, ANNOTATE_PLIES), r = new Chess(), evals = [(await analyse(r.fen())).cp]
    for (const m of moves) { r.move(m); evals.push((await analyse(r.fen())).cp) }
    g.evals = evals
  }
}

/** Targets from one Stockfish call per sample: value = its eval squashed to [−1, 1] from the side to move; policy = the sample's own target
 *  (search visits, or the opponent engine's move) blended half-and-half with Stockfish's best move. */
export async function annotateSamples(samples: Sample[], analyse: (fen: string) => Promise<Analysis>) {
  for (const s of samples) {
    const { cp, best } = await analyse(s.fen)
    s.v = +Math.tanh((s.fen.split(' ')[1] === 'w' ? cp : -cp) / 400).toFixed(3)
    if (best in s.pi) s.pi = Object.fromEntries(Object.entries(s.pi).map(([m, p]) => [m, +((1 - BLEND) * p + (m === best ? BLEND : 0)).toFixed(3)]))
  }
}

/** Average centipawn loss of the agent's own moves over the annotated plies, each clamped to [0, 1000]: lower is stronger. */
export function acpl(games: Pick<GameRecord, 'evals' | 'agentWhite'>[]) {
  const losses = games.flatMap(g => (g.evals ?? []).slice(1).flatMap((after, p) => // ply p takes evals[p] → evals[p+1]; white moves on even plies
    (p % 2 === 0) === g.agentWhite ? [Math.max(0, Math.min(1000, (p % 2 === 0 ? 1 : -1) * (g.evals![p] - after)))] : []))
  return losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : 0
}

/** Score / win / draw rates over the last 40 rated games, whoever the opponent; neutral when there are none. */
export function rates(history: Rated[]) {
  const last = history.slice(-40), n = last.length
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
 *  restore: pull state from KV on a cache miss.  play <n>: training games vs the pool and self-play → play-<n>.json (samples with Stockfish targets).
 *  train: merge the play shards into the buffer, one Adam pass.  eval <n>: rated games vs panel member n (the 4 rungs nearest the agent's rating),
 *  frozen protocol → eval-<n>.json.  publish: Elo per game against each opponent's calibrated rating, then publish. */
async function main(mode: Mode, shard: number) {
  const t0 = process.env.START ? +process.env.START * 1000 : Date.now(), ts = new Date(t0).toISOString() // the run's start (the workflow exports START)
  if (!existsSync(FILES.state) && await restore({ 'worker-state': FILES.state, weights: FILES.weights, buffer: FILES.buffer })) console.log('restored state from kv')
  if (mode === 'restore') return
  const state = read<State>(FILES.state, { run: 0, elo: TIERS[0].rating, evalHistory: [], record: {}, gamesTotal: 0, loss: 0, policyLoss: 0, valueLoss: 0 })
  const panel = nearest(state.elo, PANEL)
  const net = createNet()
  if (mode !== 'publish' && existsSync(FILES.weights)) try { await loadWeights(net, FILES.weights) } catch { console.warn('weights do not fit this architecture: starting fresh') }
  const withStockfish = async (f: (analyse: (fen: string) => Promise<Analysis>) => Promise<void>) => {
    try { const sf = await playerFor(ANNOTATOR, ENGINE); try { await f(sf.analyse) } finally { sf.quit() } } catch (e) { console.warn(`annotate: ${(e as Error).message}`) }
  }
  // one engine process per game: a timed-out engine is killed, so it must never be shared across games
  const vs = async (agent: Player, n: number, against: Tier): Promise<Game> => {
    const opp = await playerFor(against, ENGINE), agentWhite = n % 2 === 0
    try { return { ...await playGame(agentWhite ? agent : opp, agentWhite ? opp : agent, { maxPlies: MAX_PLIES, record: true }), agentWhite, against: against.name } } finally { opp.quit() }
  }
  const secs = () => Math.round((Date.now() - t0) / 1000)

  if (mode === 'play') {
    const explore = agentPlayer(net, SIMS, { noise: true })
    const selfPlay = async (): Promise<Game> => ({ ...await playGame(explore, explore, { maxPlies: MAX_PLIES, record: true }), agentWhite: true })
    const games = await playMany(GAMES, i => i % 2 ? selfPlay() : vs(explore, i / 2, trainingPool(state.elo)), Date.now() + BUDGET_MS, PARALLEL) // half vs the pool, half self-play
    const samples = games.flatMap(g => g.samples), pool = games.filter(g => g.against).length
    await withStockfish(ev => annotateSamples(samples, ev))
    writeFileSync(`play-${shard}.json`, JSON.stringify({ games: games.length, samples }))
    return console.log(`play ${shard}: ${pool} pool + ${games.length - pool} self-play games, ${samples.length} samples, ${secs()}s`)
  }

  if (mode === 'train') {
    const buffer = ReplayBuffer.load(FILES.buffer), shards = take<{ games: number; samples: Sample[] }>('play')
    for (const s of shards) { buffer.append(s.samples); state.gamesTotal += s.games }
    const { loss, policy, value, steps } = await trainEpoch(net, shuffle([...buffer.samples]).slice(0, TRAIN_SAMPLES), { batch: 256, lr: 1e-3 })
    const bench = await benchmark(net) // held-out positions: the learning curve
    Object.assign(state, { loss: +loss.toFixed(4), policyLoss: +policy.toFixed(4), valueLoss: +value.toFixed(4), steps: (state.steps ?? 0) + steps,
      benchMatch: +bench.match.toFixed(3), benchValue: +bench.value.toFixed(4) })
    writeFileSync(FILES.state, JSON.stringify(state, null, 1)); await saveWeights(net, FILES.weights); buffer.save(FILES.buffer)
    return console.log(`train: ${shards.length} shards, loss ${state.loss}, buffer ${buffer.samples.length}, benchmark match ${state.benchMatch} value ${state.benchValue}, ${secs()}s`)
  }

  if (mode === 'eval') {
    const exploit = agentPlayer(net, SIMS, { temperaturePlies: EVAL_TEMPERATURE_PLIES }), opponent = panel[shard % panel.length]
    const games: GameRecord[] = (await playMany(Math.ceil(EVAL_GAMES / SHARDS), k => vs(exploit, k, opponent), Infinity, PARALLEL)) // colour alternates by game number
      .map(g => ({ ts, tier: opponent.name, kind: 'eval', result: g.result, pgn: g.pgn, agentWhite: g.agentWhite, plies: g.plies }))
    await withStockfish(ev => annotate(games, ev))
    writeFileSync(`eval-${shard}.json`, JSON.stringify(games))
    return console.log(`eval ${shard}: ${games.length} rated games vs ${opponent.name} (${opponent.rating}), ${secs()}s`)
  }

  // games interleaved across opponents (shard order would rate them opponent by opponent)
  const shards = take<GameRecord[]>('eval'), games = Array.from({ length: Math.max(0, ...shards.map(s => s.length)) }, (_, k) => shards.flatMap(s => s[k] ?? []))
    .flat()
  for (const g of games) {
    const opponent = TIERS.find(t => t.name === g.tier)!, s = score(g as GameRecord & { result: Result }), before = state.elo
    state.elo = update(state.elo, opponent.rating, s); state.evalHistory.push({ tier: g.tier, score: s })
    const rec = (state.record[g.tier] ??= { w: 0, d: 0, l: 0 }); rec[s === 1 ? 'w' : s === 0 ? 'l' : 'd']++
    g.elo = Math.round(state.elo); g.delta = Math.round(state.elo - before)
  }
  games.reverse() // newest first
  const buffer = readFileSync(FILES.buffer, 'utf8'), { score40, win40, draw40 } = rates(state.evalHistory)
  const summary = { ts, run: ++state.run, elo: Math.round(state.elo), panel: nearest(state.elo, PANEL).map(t => t.name), ladder: TIERS.map(({ name, rating }) => ({ name, rating })),
    score40: +score40.toFixed(3), win40: +win40.toFixed(3), draw40: +draw40.toFixed(3), acpl: Math.round(acpl(games)),
    plies: Math.round(games.reduce((s, g) => s + g.plies, 0) / Math.max(1, games.length)), buffer: buffer.split('\n').filter(Boolean).length, seconds: secs(),
    gamesTotal: (state.gamesTotal += games.length), steps: state.steps ?? 0, loss: state.loss, policyLoss: state.policyLoss, valueLoss: state.valueLoss,
    benchMatch: state.benchMatch, benchValue: state.benchValue, compute: machine(t0), record: state.record }
  state.evalHistory = state.evalHistory.slice(-40)
  writeFileSync(FILES.state, JSON.stringify(state, null, 1))
  await publish(summary, games, { weights: readFileSync(FILES.weights, 'utf8'), buffer, state: readFileSync(FILES.state, 'utf8') })
  console.log(`run ${summary.run} elo ${summary.elo} panel ${summary.panel.join(',')} score40 ${summary.score40} acpl ${summary.acpl} rated ${games.length} policy ${summary.policyLoss} value ${summary.valueLoss} ${summary.seconds}s`)
}

const text = (f: string) => existsSync(f) ? readFileSync(f, 'utf8') : ''
/** The run's machine and how hard it worked (the workflow writes vmstat.log and phases.log); null when run without them (locally). */
function machine(t0: number) {
  const u = usage(text('vmstat.log'), text('phases.log'), t0 / 1000, Date.now() / 1000, os.totalmem() / 1024)
  return u && { cores: os.cpus().length, memGB: Math.round(os.totalmem() / 2 ** 30), model: os.cpus()[0]?.model.trim() ?? '', inFlight: SHARDS * PARALLEL, ...u }
}

const MODES: Mode[] = ['restore', 'play', 'train', 'eval', 'publish']
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const mode = MODES.find(m => m === process.argv[2])
  if (!mode) { console.error(`usage: worker.ts ${MODES.join('|')} [shard]`); process.exit(2) }
  main(mode, +(process.argv[3] ?? 0)).catch(e => { console.error(e); process.exit(1) })
}
