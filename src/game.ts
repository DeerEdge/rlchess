import { Chess } from 'chess.js'
import type * as tf from '@tensorflow/tfjs'
import { search } from './mcts.ts'
import { uciMoves } from './net.ts'

export type Player = (chess: Chess) => Promise<{ move: string; pi?: Map<string, number> }>
export type Sample = { fen: string; pi: Record<string, number>; z: number }
export type Result = '1-0' | '0-1' | '1/2-1/2'

export const randomPlayer: Player = async c => { const m = uciMoves(c); return { move: m[Math.floor(Math.random() * m.length)] } }

/** MCTS-driven player: Dirichlet noise optional, temperature 1 for the opening plies then greedy. */
export const agentPlayer = (net: tf.LayersModel, sims: number, opts: { noise?: boolean; temperaturePlies?: number } = {}): Player =>
  async c => {
    const ply = (c.moveNumber() - 1) * 2 + (c.turn() === 'b' ? 1 : 0)
    const temperature = ply < (opts.temperaturePlies ?? 20) ? 1 : 0
    const { move, pi } = await search(net, c, sims, { noise: opts.noise, temperature })
    return { move: move!, pi }
  }

export async function playGame(white: Player, black: Player, opts: { maxPlies: number; record: boolean }) {
  const chess = new Chess(), taken: { fen: string; pi: Record<string, number>; mover: 'w' | 'b' }[] = []
  while (!chess.isGameOver() && chess.history().length < opts.maxPlies) {
    const { move, pi } = await (chess.turn() === 'w' ? white : black)(chess)
    if (opts.record && pi) taken.push({ fen: chess.fen(), pi: Object.fromEntries(pi), mover: chess.turn() })
    chess.move({ from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] })
  }
  const winner = chess.isCheckmate() ? (chess.turn() === 'w' ? 'b' : 'w') : null // ponytail: ply cap and every other end = draw
  const result: Result = winner === 'w' ? '1-0' : winner === 'b' ? '0-1' : '1/2-1/2'
  chess.setHeader('Result', result); chess.setHeader('Date', new Date().toISOString().slice(0, 10))
  const samples: Sample[] = taken.map(t => ({ fen: t.fen, pi: t.pi, z: winner ? (t.mover === winner ? 1 : -1) : 0 }))
  return { pgn: chess.pgn(), result, samples, plies: chess.history().length }
}
