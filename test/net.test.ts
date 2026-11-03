import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createNet, predict, legalPolicy, paramCount, saveWeights, loadWeights, uciMoves } from '../src/net.ts'

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

test('uciMoves matches chess.js\'s public move list (queen promotions only) over random playouts', () => {
  for (let g = 0; g < 30; g++) {
    const c = new Chess()
    for (let i = 0; i < 120 && !c.isGameOver(); i++) {
      const want = c.moves({ verbose: true }).filter(m => !m.promotion || m.promotion === 'q').map(m => m.from + m.to + (m.promotion ?? ''))
      assert.deepEqual(uciMoves(c), want, c.fen())
      const m = want[Math.floor(Math.random() * want.length)]
      c.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] })
    }
  }
})

test('concurrent predicts share one network call and match one-at-a-time predicts', async () => {
  const net = createNet(), real = net.predict.bind(net), positions = [new Chess(), new Chess('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'), new Chess('8/8/8/4k3/8/8/4P3/4K3 b - - 0 1')]
  const single: Awaited<ReturnType<typeof predict>>[] = []
  for (const c of positions) single.push(await predict(net, c))
  let calls = 0
  net.predict = ((x, a) => { calls++; return real(x, a) }) as typeof net.predict
  const batched = await Promise.all(positions.map(c => predict(net, c)))
  assert.equal(calls, 1)
  batched.forEach((b, i) => {
    assert.ok(Math.abs(b.value - single[i].value) < 1e-5)
    assert.ok([...b.from, ...b.to].every((v, j) => Math.abs(v - [...single[i].from, ...single[i].to][j]) < 1e-4))
  })
})
