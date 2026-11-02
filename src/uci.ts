import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { Chess } from 'chess.js'
import type { Tier } from './encode.ts'
import { randomPlayer, greedyPlayer, type Player } from './game.ts'

export class EngineTimeout extends Error { constructor(name: string) { super(`${name}: no reply`) } }

export type Opts = { movetime: number; depth: number; timeoutMs?: number }

/** Spawn a UCI engine and expose it as a Player. One outstanding request at a time. */
export type Analysis = { cp: number; best: string } // cp from white's view, mate = ±1000; best = Stockfish's move (uci), '(none)' when the game is over
export type Engine = Player & { quit(): void; analyse(fen: string): Promise<Analysis> }

export async function uciPlayer(tier: Tier, { movetime, depth, timeoutMs = 5000 }: Opts): Promise<Engine> {
  const [cmd, ...args] = tier.cmd!
  const proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'ignore'] })
  const lines = createInterface({ input: proc.stdout })
  let dead: Error | null = null
  const die = () => { dead ??= new EngineTimeout(tier.name); proc.kill() }
  proc.on('error', die); proc.on('exit', () => { dead ??= new EngineTimeout(tier.name) })
  const send = (s: string) => { if (!dead) proc.stdin.write(s + '\n') }
  // A reply that arrives after its deadline would be read as the answer to the next position, so a timeout kills the engine.
  const waitFor = (re: RegExp) => new Promise<string>((resolve, reject) => {
    if (dead) return reject(dead)
    const timer = setTimeout(() => { lines.off('line', onLine); die(); reject(dead) }, timeoutMs)
    const onLine = (l: string) => { const m = l.match(re); if (m) { clearTimeout(timer); lines.off('line', onLine); resolve(m[1] ?? l) } }
    lines.on('line', onLine)
  })
  send('uci'); await waitFor(/^uciok/)
  for (const [k, v] of Object.entries(tier.options ?? {})) send(`setoption name ${k} value ${v}`)
  send('isready'); await waitFor(/^readyok/)
  let score = 0 // last "info … score" seen, from the side to move
  lines.on('line', l => { const m = l.match(/ score (cp|mate) (-?\d+)/); if (m) score = m[1] === 'cp' ? +m[2] : Math.sign(+m[2] || -1) * 1000 })
  const player = (async (chess: Chess) => {
    send(`position fen ${chess.fen()}`); send(`go movetime ${movetime} depth ${depth}`)
    return { move: await waitFor(/^bestmove (\S+)/), expert: true }
  }) as Engine
  player.analyse = async fen => {
    send(`position fen ${fen}`); send(`go depth ${depth}`)
    const best = await waitFor(/^bestmove (\S+)/)
    return { cp: Math.max(-1000, Math.min(1000, fen.split(' ')[1] === 'w' ? score : -score)), best }
  }
  player.quit = () => { send('quit'); proc.kill() }
  proc.stdin.on('error', () => {})
  return player
}

/** The tier's Player: a spawned engine (at the rung's own depth, diluted with random moves if it says so), or one of the internal movers (random, greedy). */
export async function playerFor(tier: Tier, opts: Opts): Promise<Engine> {
  if (!tier.cmd) return Object.assign(tier.name === 'greedy' ? greedyPlayer : randomPlayer, { quit() {}, analyse: async () => ({ cp: 0, best: '(none)' }) })
  const engine = await uciPlayer(tier, { ...opts, depth: tier.depth ?? opts.depth })
  if (!tier.random) return engine
  return Object.assign(async (c: Chess) => Math.random() < tier.random! ? randomPlayer(c) : engine(c), { quit: engine.quit, analyse: engine.analyse }) as Engine
}
