import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { Chess } from 'chess.js'
import type { Tier } from './encode.ts'
import { randomPlayer, type Player } from './game.ts'

export class EngineTimeout extends Error { constructor(name: string) { super(`${name}: no reply`) } }

export type Opts = { movetime: number; depth: number; timeoutMs?: number }

/** Spawn a UCI engine and expose it as a Player. One outstanding request at a time. */
export type Engine = Player & { quit(): void; evaluate(fen: string): Promise<number> }

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
    return { move: await waitFor(/^bestmove (\S+)/) }
  }) as Engine
  /** Centipawns from white's view, mate = ±1000. */
  player.evaluate = async fen => {
    send(`position fen ${fen}`); send(`go depth ${depth}`); await waitFor(/^bestmove/)
    return Math.max(-1000, Math.min(1000, fen.split(' ')[1] === 'w' ? score : -score))
  }
  player.quit = () => { send('quit'); proc.kill() }
  proc.stdin.on('error', () => {})
  return player
}

/** The tier's Player: internal random mover, or a spawned engine. */
export const playerFor = (tier: Tier, opts: Opts): Promise<Engine> =>
  tier.cmd ? uciPlayer(tier, opts) : Promise.resolve(Object.assign(randomPlayer, { quit() {}, evaluate: async () => 0 }))
