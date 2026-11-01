import { test } from 'node:test'
import assert from 'node:assert/strict'
import { elo, promote } from '../src/elo.ts'

test('elo formula with clamping', () => {
  assert.equal(elo(0.5, 1500), 1500)
  assert.ok(Math.abs(elo(0.99, 1500) - 2298) < 1)
  assert.equal(elo(1, 1500), elo(0.99, 1500))
  assert.equal(elo(0, 1500), elo(0.01, 1500))
})

const hist = (n: number, score: number, tier = 'random') => Array.from({ length: n }, () => ({ tier, score }))
const next = (cur: string) => promote(hist(40, 1, cur), cur)

test('promote needs 40 eval games of the current tier at mean >= 0.55', () => {
  assert.equal(promote(hist(39, 1), 'random'), 'random')
  assert.equal(promote(hist(40, 0.56), 'random'), 'sunfish')
  assert.equal(promote(hist(40, 0.54), 'random'), 'random')
  assert.equal(promote([...hist(39, 1), ...hist(1, 1, 'lozza')], 'random'), 'random')
  assert.equal(promote([...hist(10, 0, 'random'), ...hist(40, 0.6, 'random')], 'random'), 'sunfish', 'only the last 40 count')
  assert.equal(next('stockfish-20'), 'stockfish-20', 'top of the ladder stays')
})
