import { test } from 'node:test'
import assert from 'node:assert/strict'
import { update, expected, nearest, fit, type Match } from '../src/elo.ts'

test('standard Elo update, K = 32', () => {
  assert.equal(update(400, 400, 1), 416)
  assert.equal(update(400, 400, 0.5), 400)
  assert.ok(Math.abs(update(400, 800, 0) - 397.1) < 0.1, 'losing to a much stronger opponent costs little')
  assert.ok(Math.abs(update(400, 800, 1) - 429.1) < 0.1, 'beating one pays a lot')
})

test('nearest picks the n rungs closest in rating, in rating order', () => {
  const ladder = [100, 300, 500, 700, 900, 1100].map((rating, i) => ({ name: `r${i}`, rating }))
  assert.deepEqual(nearest(640, 4, ladder).map(t => t.name), ['r1', 'r2', 'r3', 'r4'])
  assert.deepEqual(nearest(-50, 2, ladder).map(t => t.name), ['r0', 'r1'], 'clamps at the bottom')
})

test('fit recovers ratings from results, anchored', () => {
  const truth: Record<string, number> = { a: 1000, b: 1200, c: 1450, d: 1600 }
  const matches: Match[] = []
  for (const [x, y] of [['a', 'b'], ['b', 'c'], ['c', 'd'], ['a', 'c'], ['b', 'd']])
    for (let g = 0; g < 400; g++) matches.push({ a: x, b: y, score: expected(truth[x], truth[y]) }) // the average result, game by game
  const r = fit(matches, ['a', 1000])
  for (const n of Object.keys(truth)) assert.ok(Math.abs(r[n] - truth[n]) < 15, `${n}: ${r[n]} vs ${truth[n]}`)
})

test('fit stays finite when one side sweeps', () => {
  const r = fit(Array.from({ length: 40 }, () => ({ a: 'weak', b: 'strong', score: 0 })), ['strong', 1000])
  assert.ok(Number.isFinite(r.weak) && r.weak < 600)
})
