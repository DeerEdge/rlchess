import tf from './tf.ts'
import type { Chess } from 'chess.js'
import { readFile, writeFile } from 'node:fs/promises'
import { encode, moveIndex, PLANES } from './encode.ts'

/** Small conv tower on the 8×8×18 planes: 4 × (3×3, 64 filters) → policy head (1×1 conv → 128 from/to logits) + value head (1×1 conv → 64 → tanh).
 *  142k params; 2.7 ms/predict, 0.2 s per 256-batch on the native CPU backend. */
export function createNet(): tf.LayersModel {
  const input = tf.input({ shape: [8, 8, PLANES] })
  let x: tf.SymbolicTensor = input
  for (let i = 0; i < 4; i++) x = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x) as tf.SymbolicTensor
  const head = (filters: number) => tf.layers.flatten().apply(tf.layers.conv2d({ filters, kernelSize: 1, activation: 'relu' }).apply(x)) as tf.SymbolicTensor
  const policy = tf.layers.dense({ units: 128, name: 'policy' }).apply(head(2)) as tf.SymbolicTensor
  const value = tf.layers.dense({ units: 1, activation: 'tanh', name: 'value' }).apply(tf.layers.dense({ units: 64, activation: 'relu' }).apply(head(1)) as tf.SymbolicTensor) as tf.SymbolicTensor
  return tf.model({ inputs: input, outputs: [policy, value] })
}

export const paramCount = (net: tf.LayersModel) => net.countParams()

/** Planes [PLANES*64] per sample (plane-major) → input tensor [n, 8, 8, PLANES] (channels last). */
export const toInput = (planes: Float32Array[]) => tf.tidy(() => tf.tensor(planes.flatMap(p => [...p]), [planes.length, PLANES, 8, 8]).transpose([0, 2, 3, 1]))

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

/** Weights as JSON shapes + base64 float32 (~460 KB), not a float-per-token JSON array. */
export async function saveWeights(net: tf.LayersModel, path: string) {
  const ws = net.getWeights(), data = await Promise.all(ws.map(w => w.data() as Promise<Float32Array>))
  const buf = Buffer.concat(data.map(d => Buffer.from(d.buffer, d.byteOffset, d.byteLength)))
  await writeFile(path, JSON.stringify({ shapes: ws.map(w => w.shape), data: buf.toString('base64') }))
}

export async function loadWeights(net: tf.LayersModel, path: string) {
  const { shapes, data } = JSON.parse(await readFile(path, 'utf8')) as { shapes: number[][]; data: string }
  const all = new Float32Array(new Uint8Array(Buffer.from(data, 'base64')).buffer)
  let off = 0
  net.setWeights(shapes.map(s => { const n = s.reduce((a, b) => a * b, 1); const t = tf.tensor(all.slice(off, off + n), s); off += n; return t }))
}
