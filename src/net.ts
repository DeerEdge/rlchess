import * as tf from '@tensorflow/tfjs'
import type { Chess } from 'chess.js'
import { readFile, writeFile } from 'node:fs/promises'
import { encode, moveIndex, PLANES } from './encode.ts'

tf.env().set('IS_NODE', true)

/** Tiny policy/value MLP on the flattened planes: 1152 → 64 → 64 → policy 128 (from/to logits) + value tanh.
 *  ponytail: dense, not conv — tfjs's pure-JS conv backprop is ~40× slower; 86k params, 0.8 ms/predict. */
export function createNet(): tf.LayersModel {
  const input = tf.input({ shape: [PLANES * 64] })
  let x: tf.SymbolicTensor = input
  for (const units of [64, 64]) x = tf.layers.dense({ units, activation: 'relu' }).apply(x) as tf.SymbolicTensor
  const policy = tf.layers.dense({ units: 128, name: 'policy' }).apply(x) as tf.SymbolicTensor
  const value = tf.layers.dense({ units: 1, activation: 'tanh', name: 'value' }).apply(x) as tf.SymbolicTensor
  return tf.model({ inputs: input, outputs: [policy, value] })
}

export const paramCount = (net: tf.LayersModel) => net.countParams()

/** Planes [PLANES*64] per sample → input tensor [n, PLANES*64]. */
export const toInput = (planes: Float32Array[]) => tf.tensor(planes.flatMap(p => [...p]), [planes.length, PLANES * 64])

/** Policy output = 64 from-square logits followed by 64 to-square logits. */
export const splitPolicy = (logits: Float32Array | number[]) => ({ from: Float32Array.from(logits.slice(0, 64)), to: Float32Array.from(logits.slice(64, 128)) })

export async function predict(net: tf.LayersModel, chess: Chess) {
  const [p, v] = tf.tidy(() => net.predict(toInput([encode(chess)])) as tf.Tensor[])
  const [logits, value] = await Promise.all([p.data(), v.data()])
  p.dispose(); v.dispose()
  return { ...splitPolicy(logits as Float32Array), value: value[0] }
}

/** Softmax over legal uci moves of from[f] + to[t]; `flip` when black to move (indices mirror the encoding). */
export function legalPolicy(from: Float32Array, to: Float32Array, moves: string[], flip = false): Map<string, number> {
  const scores = moves.map(m => { const { from: f, to: t } = moveIndex(m, flip); return from[f] + to[t] })
  const max = Math.max(...scores), exps = scores.map(s => Math.exp(s - max)), z = exps.reduce((a, b) => a + b, 0)
  return new Map(moves.map((m, i) => [m, exps[i] / z]))
}

/** Legal moves as uci, queen-only promotions (underpromotion dropped on purpose — tiny net). */
export const uciMoves = (chess: Chess) =>
  chess.moves({ verbose: true }).filter(m => !m.promotion || m.promotion === 'q').map(m => m.from + m.to + (m.promotion ?? ''))

export async function saveWeights(net: tf.LayersModel, path: string) {
  const ws = net.getWeights()
  const data = await Promise.all(ws.map(w => w.data()))
  await writeFile(path, JSON.stringify({ shapes: ws.map(w => w.shape), data: data.map(d => [...d]) }))
}

export async function loadWeights(net: tf.LayersModel, path: string) {
  const { shapes, data } = JSON.parse(await readFile(path, 'utf8')) as { shapes: number[][]; data: number[][] }
  net.setWeights(shapes.map((s, i) => tf.tensor(data[i], s)))
}
