import { test } from 'node:test'
import tf from '../src/tf.ts'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { createNet } from '../src/net.ts'
import { trainEpoch } from '../src/train.ts'
import type { Sample } from '../src/game.ts'

test('one epoch on a repeated sample lowers the loss', async () => {
  const net = createNet()
  const s: Sample = { fen: new Chess().fen(), pi: { e2e4: 0.7, d2d4: 0.3 }, z: 1 }
  const samples = Array.from({ length: 64 }, () => s)
  const a = await trainEpoch(net, samples, { batch: 32, lr: 1e-3 })
  const b = await trainEpoch(net, samples, { batch: 32, lr: 1e-3 })
  assert.ok(Number.isFinite(a.loss))
  assert.ok(b.loss < a.loss, `${b.loss} !< ${a.loss}`)
  assert.ok(Math.abs(a.policy + a.value - a.loss) < 1e-4, 'loss = policy + value, reported separately')
})

test('an empty sample set trains nothing and reports loss 0', async () => {
  assert.deepEqual(await trainEpoch(createNet(), [], { batch: 32, lr: 1e-3 }), { loss: 0, policy: 0, value: 0 })
})

test('a passed-in optimizer carries its state across calls', async () => {
  const net = createNet(), opt = tf.train.adam(1e-3)
  const samples: Sample[] = Array.from({ length: 64 }, () => ({ fen: new Chess().fen(), pi: { e2e4: 1, d2d4: 0 }, z: 0 }))
  await trainEpoch(net, samples, { batch: 32, lr: 1e-3, opt })
  await trainEpoch(net, samples, { batch: 32, lr: 1e-3, opt })
  assert.equal(opt.iterations, 4)
})
