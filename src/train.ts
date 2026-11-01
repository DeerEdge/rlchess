import tf from './tf.ts'
import { Chess } from 'chess.js'
import { encode, moveIndex } from './encode.ts'
import { toInput } from './net.ts'
import type { Sample } from './game.ts'
import { shuffle } from './buffer.ts'

/** Policy targets as padded per-sample (from, to) index lists over the sample's own move set. */
function targets(samples: Sample[]) {
  const L = Math.max(...samples.map(s => Object.keys(s.pi).length))
  const from: number[][] = [], to: number[][] = [], mask: number[][] = [], pi: number[][] = []
  for (const s of samples) {
    const flip = s.fen.split(' ')[1] === 'b', moves = Object.entries(s.pi)
    const row = (f: (m: [string, number]) => number, pad: number) => [...moves.map(f), ...Array(L - moves.length).fill(pad)]
    from.push(row(([m]) => moveIndex(m, flip).from, 0)); to.push(row(([m]) => moveIndex(m, flip).to, 0))
    mask.push(row(() => 1, 0)); pi.push(row(([, p]) => p, 0))
  }
  return { from: tf.tensor2d(from, undefined, 'int32'), to: tf.tensor2d(to, undefined, 'int32'), mask: tf.tensor2d(mask), pi: tf.tensor2d(pi) }
}

/** One pass over `samples`: cross-entropy on legal-move policy + MSE on value (engine eval if present, else outcome). Returns mean loss. */
export async function trainEpoch(net: tf.LayersModel, samples: Sample[], { batch, lr, onBatch }: { batch: number; lr: number; onBatch?: (done: number, total: number, loss: number) => void }) {
  if (!samples.length) return { loss: 0 }
  const opt = tf.train.adam(lr), losses: number[] = []
  const order = shuffle(samples.map((_, i) => i))
  for (let i = 0; i < order.length; i += batch) {
    const chunk = order.slice(i, i + batch).map(j => samples[j])
    const x = toInput(chunk.map(s => encode(new Chess(s.fen)))), t = targets(chunk), z = tf.tensor2d(chunk.map(s => [s.v ?? s.z])) // engine value when annotated, else the outcome
    const loss = opt.minimize(() => {
      const [p, v] = net.apply(x) as tf.Tensor[]
      const gf = tf.gather(p.slice([0, 0], [-1, 64]), t.from, 1, 1)
      const gt = tf.gather(p.slice([0, 64], [-1, 64]), t.to, 1, 1)
      const logits = gf.add(gt).add(t.mask.sub(1).mul(1e9))
      const policyLoss = tf.logSoftmax(logits).mul(t.pi).sum(1).neg().mean()
      return policyLoss.add(tf.losses.meanSquaredError(z, v)) as tf.Scalar
    }, true)!
    losses.push((await loss.data())[0])
    onBatch?.(losses.length, Math.ceil(order.length / batch), losses.at(-1)!)
    tf.dispose([x, z, loss, ...Object.values(t)])
  }
  opt.dispose()
  return { loss: losses.reduce((a, b) => a + b, 0) / losses.length }
}
