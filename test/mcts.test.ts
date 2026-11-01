import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { createNet } from '../src/net.ts'
import { search } from '../src/mcts.ts'

const net = createNet()

test('finds mate in one with an untrained net', async () => {
  const chess = new Chess('6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1')
  const { move } = await search(net, chess, 60, { temperature: 0 })
  assert.equal(move, 'a1a8')
})

test('single legal move is returned with a one-entry pi', async () => {
  const chess = new Chess('7k/8/8/8/8/8/5q2/K7 w - - 0 1') // only Ka1-b1... check: Kb1 legal? queen f2 covers b2; a2 covered. Ka1→b1 only.
  const { move, pi } = await search(net, chess, 10)
  assert.equal(chess.moves().length, 1)
  assert.equal(move, 'a1b1')
  assert.equal(pi.size, 1)
})

test('position with no legal moves returns null without throwing', async () => {
  const chess = new Chess('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1') // stalemate
  const { move, pi } = await search(net, chess, 10)
  assert.equal(move, null)
  assert.equal(pi.size, 0)
})
