import { test } from 'node:test'
import assert from 'node:assert/strict'
import { update, promote } from '../src/elo.ts'

test('standard Elo update, K = 32', () => {
  assert.equal(update(400, 400, 1), 416)
  assert.equal(update(400, 400, 0.5), 400)
  assert.ok(Math.abs(update(400, 800, 0) - 397.1) < 0.1, 'losing to a much stronger opponent costs little')
  assert.ok(Math.abs(update(400, 800, 1) - 429.1) < 0.1, 'beating one pays a lot')
})

const hist = (n: number, score: number, tier = 'random') => Array.from({ length: n }, () => ({ tier, score }))
const next = (cur: string) => promote(hist(40, 1, cur), cur)

test('promote needs 40 eval games of the current tier at mean >= 0.55', () => {
  assert.equal(promote(hist(39, 1), 'random'), 'random')
  assert.equal(promote(hist(40, 0.56), 'random'), 'greedy')
  assert.equal(promote(hist(40, 0.54), 'random'), 'random')
  assert.equal(promote([...hist(39, 1), ...hist(1, 1, 'lozza')], 'random'), 'random')
  assert.equal(promote([...hist(10, 0, 'random'), ...hist(40, 0.6, 'random')], 'random'), 'greedy', 'only the last 40 count')
  assert.equal(next('stockfish-20'), 'stockfish-20', 'top of the ladder stays')
})
