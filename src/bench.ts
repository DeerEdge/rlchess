import { readFileSync } from 'node:fs'
import { Chess } from 'chess.js'
import tf from './tf.ts'
import { encode } from './encode.ts'
import { toInput, splitPolicy, legalPolicy, uciMoves } from './net.ts'

export type Labelled = { fen: string; best: string; v: number } // v: Stockfish's eval squashed to [−1, 1], from the side to move
/** 2,000 Lichess positions never trained on: lines 3,000,001–3,002,000 of the evaluation database (pretraining read the first 3M). */
export const BENCH: Labelled[] = readFileSync(new URL('../bench.jsonl', import.meta.url), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l))

/** The raw net (no search) on held-out positions: how often its top move is Stockfish's best, and its value error (MSE).
 *  A learning curve with no opponents and no dice in it. */
export async function benchmark(net: tf.LayersModel, positions: Labelled[] = BENCH) {
  let hits = 0, err = 0
  for (let i = 0; i < positions.length; i += 256) {
    const chunk = positions.slice(i, i + 256), boards = chunk.map(p => new Chess(p.fen))
    const [p, v] = tf.tidy(() => net.predict(toInput(boards.map(encode))) as tf.Tensor[])
    const [logits, values] = await Promise.all([p.data(), v.data()]) as Float32Array[]
    p.dispose(); v.dispose()
    chunk.forEach((pos, k) => {
      const { from, to } = splitPolicy(logits.subarray(k * 128, (k + 1) * 128))
      const pi = legalPolicy(from, to, uciMoves(boards[k]), boards[k].turn() === 'b')
      hits += +([...pi].reduce((a, b) => (b[1] > a[1] ? b : a))[0] === pos.best); err += (values[k] - pos.v) ** 2
    })
  }
  return { match: hits / positions.length, value: err / positions.length }
}
