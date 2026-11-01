import * as tf from '@tensorflow/tfjs'
import type { Chess } from 'chess.js'
import { readFile, writeFile } from 'node:fs/promises'
import { encode, moveIndex, PLANES } from './encode.ts'

tf.env().set('IS_NODE', true)

/** Tiny policy/value net: 3 small convs; policy = from-square + to-square logits (128), value = tanh scalar. */
export function createNet(): tf.LayersModel {
  const input = tf.input({ shape: [8, 8, PLANES] })
  let x: tf.SymbolicTensor = input
  for (const filters of [16, 32, 32]) x = tf.layers.conv2d({ filters, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x) as tf.SymbolicTensor
  const policy = tf.layers.flatten({ name: 'policy' }).apply(tf.layers.conv2d({ filters: 2, kernelSize: 1 }).apply(x)) as tf.SymbolicTensor
  let v = tf.layers.flatten().apply(tf.layers.conv2d({ filters: 1, kernelSize: 1, activation: 'relu' }).apply(x)) as tf.SymbolicTensor
  v = tf.layers.dense({ units: 32, activation: 'relu' }).apply(v) as tf.SymbolicTensor
  const value = tf.layers.dense({ units: 1, activation: 'tanh', name: 'value' }).apply(v) as tf.SymbolicTensor
  return tf.model({ inputs: input, outputs: [policy, value] })
}

export const paramCount = (net: tf.LayersModel) => net.countParams()

/** Planes [PLANES*64] → NHWC tensor [n,8,8,PLANES]. */
export const toInput = (planes: Float32Array[]) =>
  tf.tensor(planes.flatMap(p => [...p]), [planes.length, PLANES, 8, 8]).transpose([0, 2, 3, 1])

/** Split the flattened [8,8,2] policy output (interleaved per square) into from/to heads. */
export const splitPolicy = (logits: Float32Array | number[]) => {
  const from = new Float32Array(64), to = new Float32Array(64)
  for (let s = 0; s < 64; s++) { from[s] = logits[2 * s]; to[s] = logits[2 * s + 1] }
  return { from, to }
}

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
