import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createNet, predict, legalPolicy, paramCount, saveWeights, loadWeights } from '../src/net.ts'

test('net stays under the 200k parameter budget', () => {
  const n = paramCount(createNet())
  console.log('params:', n)
  assert.ok(n < 200_000)
})

test('legalPolicy is a distribution over the given moves', () => {
  const from = new Float32Array(64).fill(0), to = new Float32Array(64).fill(0)
  from[12] = 2; to[28] = 1
  const pi = legalPolicy(from, to, ['e2e4', 'e2e3', 'd2d4'])
  assert.equal(pi.size, 3)
  const total = [...pi.values()].reduce((s, v) => s + v, 0)
  assert.ok(Math.abs(total - 1) < 1e-6)
  assert.ok([...pi.values()].every(v => v > 0))
  assert.ok(pi.get('e2e4')! > pi.get('d2d4')!)
})

test('predict returns a value in [-1, 1] and 64-length heads', async () => {
  const { from, to, value } = await predict(createNet(), new Chess())
  assert.equal(from.length, 64); assert.equal(to.length, 64)
  assert.ok(value >= -1 && value <= 1)
})

test('weights round-trip through json', async () => {
  const a = createNet(), path = mkdtempSync(tmpdir() + '/w') + '/w.json'
  await saveWeights(a, path)
  const b = createNet(); await loadWeights(b, path)
  const pa = await predict(a, new Chess()), pb = await predict(b, new Chess())
  assert.deepEqual([...pa.from], [...pb.from]); assert.equal(pa.value, pb.value)
})
