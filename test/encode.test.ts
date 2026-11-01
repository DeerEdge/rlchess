import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { encode, moveIndex, PLANES, TIERS } from '../src/encode.ts'

const plane = (x: Float32Array, p: number) => x.subarray(p * 64, p * 64 + 64)
const sum = (a: Float32Array) => a.reduce((s, v) => s + v, 0)

test('startpos has 32 piece bits and side-to-move plane all ones', () => {
  const x = encode(new Chess())
  assert.equal(x.length, PLANES * 64)
  let pieces = 0
  for (let p = 0; p < 12; p++) pieces += sum(plane(x, p))
  assert.equal(pieces, 32)
  assert.equal(sum(plane(x, 12)), 64)
})

test('black to move is flipped so my pawns sit on rank 2', () => {
  const c = new Chess(); c.move('e4')
  const x = encode(c)
  assert.equal(plane(x, 0)[8], 1)   // a7 black pawn → a2 after flip, plane 0 = my pawns
  assert.equal(plane(x, 0)[36], 0)  // e4 white pawn is "theirs", not in plane 0
  assert.equal(plane(x, 6)[36], 1)  // e4 (rank idx 3) → rank idx 4 after flip = index 36, plane 6 = their pawns
})

test('moveIndex maps uci squares to 0-63 (a1 = 0), with optional flip', () => {
  assert.deepEqual(moveIndex('e2e4'), { from: 12, to: 28 })
  assert.deepEqual(moveIndex('e7e5', true), { from: 12, to: 28 })
})

test('tiers are ordered by rating with random first', () => {
  assert.equal(TIERS[0].name, 'random')
  assert.ok(TIERS.length >= 20)
})
