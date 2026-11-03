import { test } from 'node:test'
import assert from 'node:assert/strict'
import { toSample } from '../src/pretrain.ts'

const line = (fen: string, evals: object[]) => JSON.stringify({ fen, evals })

test('the deepest eval\'s first move is a one-hot target over every legal move; value is its score from the side to move', () => {
  const s = toSample(line('7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -', [
    { pvs: [{ cp: 300, line: 'h8d8 d1e1' }], depth: 20 },
    { pvs: [{ cp: 69, line: 'f7g7 e6e2' }, { cp: 163, line: 'h8d8 d1e1' }], depth: 36 }]))!
  assert.equal(s.fen, '7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - - 0 1')
  assert.equal(s.pi.f7g7, 1)
  assert.equal(Object.values(s.pi).reduce((a: number, b: number) => a + b, 0), 1)
  assert.ok(Object.keys(s.pi).length > 10 && 'h8d8' in s.pi, 'every legal move is in the target')
  assert.equal(s.v, +(-Math.tanh(69 / 400)).toFixed(3), 'cp is white\'s view; black is to move')
})

test('mate scores saturate like the loop\'s (±1000 cp), from the side to move', () => {
  const s = toSample(line('6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -', [{ pvs: [{ mate: 1, line: 'a1a8' }], depth: 245 }]))!
  assert.equal(s.v, +Math.tanh(1000 / 400).toFixed(3))
  const t = toSample(line('6k1/5ppp/8/8/8/8/r4PPP/6K1 w - -', [{ pvs: [{ mate: -1, line: 'h2h3' }], depth: 245 }]))!
  assert.equal(t.v, -s.v)
})

test('Chess960-style castling (king takes own rook) becomes standard uci; a rook move on the same squares does not', () => {
  assert.equal(Object.entries(toSample(line('r3k2r/8/8/8/8/8/8/R3K2R w KQkq -', [{ pvs: [{ cp: 0, line: 'e1h1' }], depth: 30 }]))!.pi).find(([, p]) => p)![0], 'e1g1')
  assert.equal(Object.entries(toSample(line('r3k2r/8/8/8/8/8/8/R3K2R b KQkq -', [{ pvs: [{ cp: 0, line: 'e8a8' }], depth: 30 }]))!.pi).find(([, p]) => p)![0], 'e8c8')
  assert.equal(Object.entries(toSample(line('3k4/8/8/8/8/8/8/K3R3 w - -', [{ pvs: [{ cp: 0, line: 'e1h1' }], depth: 30 }]))!.pi).find(([, p]) => p)![0], 'e1h1')
})

test('an underpromotion best move (outside the net\'s move set) is skipped', () => {
  assert.equal(toSample(line('8/P6k/8/8/8/8/8/K7 w - -', [{ pvs: [{ cp: 0, line: 'a7a8n' }], depth: 30 }])), null)
})
