import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { createNet, uciMoves } from '../src/net.ts'
import { benchmark, BENCH } from '../src/bench.ts'

test('benchmark counts a forced move as a match and reports the value error', async () => {
  const r = await benchmark(createNet(), [{ fen: '7k/8/8/8/8/8/5q2/K7 w - - 0 1', best: 'a1b1', v: -1 }])
  assert.equal(r.match, 1)
  assert.ok(r.value >= 0 && r.value <= 4)
})

test('the benchmark set is 2,000-odd held-out positions, each with a legal best move and a value in [-1, 1]', () => {
  assert.ok(BENCH.length > 1990 && BENCH.length <= 2000, String(BENCH.length))
  for (const p of BENCH) { assert.ok(uciMoves(new Chess(p.fen)).includes(p.best), p.fen); assert.ok(Math.abs(p.v) <= 1) }
})
