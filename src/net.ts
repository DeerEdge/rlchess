import tf from './tf.ts'
import type { Chess } from 'chess.js'
import { readFile, writeFile } from 'node:fs/promises'
import { encode, moveIndex, PLANES } from './encode.ts'

/** Small conv tower on the 8×8×18 planes: 4 × (3×3, 64 filters) → policy head (1×1 conv → 128 from/to logits) + value head (1×1 conv → 64 → tanh).
 *  142k params. */
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

type Prediction = { from: Float32Array; to: Float32Array; value: number }
type Pending = { x: Float32Array; resolve: (p: Prediction) => void; reject: (e: unknown) => void }
const queued = new Map<tf.LayersModel, Pending[]>()

/** Positions asked for in the same tick (concurrent games' searches) go through the net as one batch: ~0.8 ms a position at 16 vs 3.9 ms alone. */
export function predict(net: tf.LayersModel, chess: Chess): Promise<Prediction> {
  return new Promise((resolve, reject) => {
    let q = queued.get(net)
    if (!q) { queued.set(net, q = []); setImmediate(() => flush(net)) }
    q.push({ x: encode(chess), resolve, reject })
  })
}

async function flush(net: tf.LayersModel) {
  const q = queued.get(net)!; queued.delete(net)
  try {
    const [p, v] = tf.tidy(() => net.predict(toInput(q.map(r => r.x))) as tf.Tensor[])
    const [logits, values] = await Promise.all([p.data(), v.data()]) as Float32Array[]
    p.dispose(); v.dispose()
    q.forEach((r, i) => r.resolve({ ...splitPolicy(logits.subarray(i * 128, (i + 1) * 128)), value: values[i] }))
  } catch (e) { q.forEach(r => r.reject(e)) }
}

/** Softmax over legal uci moves of from[f] + to[t]; `flip` when black to move (indices mirror the encoding). */
export function legalPolicy(from: Float32Array, to: Float32Array, moves: string[], flip = false): Map<string, number> {
  const scores = moves.map(m => { const { from: f, to: t } = moveIndex(m, flip); return from[f] + to[t] })
  const max = Math.max(...scores), exps = scores.map(s => Math.exp(s - max)), z = exps.reduce((a, b) => a + b, 0)
  return new Map(moves.map((m, i) => [m, exps[i] / z]))
}

type Internal = { _moves(o: { legal: boolean }): { from: number; to: number; promotion?: string }[] }
const algebraic = (sq: number) => 'abcdefgh'[sq & 15] + (8 - (sq >> 4)) // chess.js's 0x88 board
/** Legal moves as uci, queen-only promotions (underpromotion dropped on purpose — tiny net).
 *  ponytail: chess.js's internal generator, 40× faster than moves({ verbose }) (which builds SAN and FENs per move); a test pins it to the public list. */
export const uciMoves = (chess: Chess) => (chess as unknown as Internal)._moves({ legal: true })
  .filter(m => !m.promotion || m.promotion === 'q').map(m => algebraic(m.from) + algebraic(m.to) + (m.promotion ?? ''))

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
