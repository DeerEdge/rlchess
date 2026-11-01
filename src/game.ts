import { Chess } from 'chess.js'
import type tf from './tf.ts'
import { search } from './mcts.ts'
import { uciMoves } from './net.ts'

export type Player = (chess: Chess) => Promise<{ move: string; pi?: Map<string, number>; expert?: boolean }>
export type Sample = { fen: string; pi: Record<string, number>; z: number; v?: number } // v: engine value target from the side to move, when annotated
export type Result = '1-0' | '0-1' | '1/2-1/2'

export const randomPlayer: Player = async c => { const m = uciMoves(c); return { move: m[Math.floor(Math.random() * m.length)] } }

const VALUE: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }
/** Ladder rung between random and an engine: mate in one if there is one, else the most valuable capture, else random. */
export const greedyPlayer: Player = async c => {
  const moves = c.moves({ verbose: true }).filter(m => !m.promotion || m.promotion === 'q')
  const mate = moves.find(m => { const t = new Chess(c.fen()); t.move(m); return t.isCheckmate() })
  const captures = moves.filter(m => m.captured).sort((a, b) => VALUE[b.captured!] - VALUE[a.captured!])
  const m = mate ?? captures[0]
  return m ? { move: m.from + m.to + (m.promotion ?? '') } : randomPlayer(c)
}

/** Material balance in pawns from the side to move's view, for shaping capped games. */
const material = (c: Chess) => c.board().flat().reduce((s, p) => p ? s + (p.color === c.turn() ? 1 : -1) * VALUE[p.type] : s, 0)

/** MCTS-driven player: Dirichlet noise optional, temperature 1 for the opening plies then greedy. */
export const agentPlayer = (net: tf.LayersModel, sims: number, opts: { noise?: boolean; temperaturePlies?: number } = {}): Player =>
  async c => {
    const ply = (c.moveNumber() - 1) * 2 + (c.turn() === 'b' ? 1 : 0)
    const temperature = ply < (opts.temperaturePlies ?? 20) ? 1 : 0
    const { move, pi } = await search(net, c, sims, { noise: opts.noise, temperature })
    return { move: move!, pi }
  }

export async function playGame(white: Player, black: Player, opts: { maxPlies: number; record: boolean; onMove?: (chess: Chess) => void; fen?: string }) {
  const chess = new Chess(opts.fen), taken: { fen: string; pi: Record<string, number>; mover: 'w' | 'b' }[] = []
  while (!chess.isGameOver() && chess.history().length < opts.maxPlies) {
    const { move, pi, expert } = await (chess.turn() === 'w' ? white : black)(chess)
    // the agent's search distribution, or an engine's move as a one-hot expert target; random/greedy movers teach nothing
    const target = pi ? Object.fromEntries([...pi].map(([m, p]) => [m, +p.toFixed(3)])) : expert ? Object.fromEntries(uciMoves(chess).map(m => [m, +(m === move)])) : null
    if (opts.record && target) taken.push({ fen: chess.fen(), pi: target, mover: chess.turn() })
    chess.move({ from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] })
    opts.onMove?.(chess)
  }
  const winner = chess.isCheckmate() ? (chess.turn() === 'w' ? 'b' : 'w') : null // ponytail: ply cap and every other end = draw
  const result: Result = winner === 'w' ? '1-0' : winner === 'b' ? '0-1' : '1/2-1/2'
  chess.setHeader('Result', result); chess.setHeader('Date', new Date().toISOString().slice(0, 10))
  // a capped game is a draw for rating, but for training it is scored by material (±0.5 max) so "being up material" is still a signal
  const capped = !chess.isGameOver(), edge = Math.max(-0.5, Math.min(0.5, material(chess) / 10)) * (chess.turn() === 'w' ? 1 : -1) // white's view
  const samples: Sample[] = taken.map(t => ({ fen: t.fen, pi: t.pi, z: winner ? (t.mover === winner ? 1 : -1) : capped ? (t.mover === 'w' ? edge : -edge) : 0 }))
  const pgn = chess.pgn().replace(/^\[.*\]\n?/gm, '').trim() // movetext only: the tag pairs are all "?" and cost more bytes than the moves
  return { pgn, result, samples, plies: chess.history().length }
}
