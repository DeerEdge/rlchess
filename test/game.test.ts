import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { playGame, randomPlayer, type Player } from '../src/game.ts'
import { ReplayBuffer } from '../src/buffer.ts'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'

test('random vs random stops at the ply cap as a draw with a parseable pgn', async () => {
  const g = await playGame(randomPlayer, randomPlayer, { maxPlies: 20, record: false })
  assert.ok(['1-0', '0-1', '1/2-1/2'].includes(g.result))
  assert.doesNotThrow(() => new Chess().loadPgn(g.pgn))
  assert.equal(g.samples.length, 0)
})

test('samples carry z from each mover\'s view', async () => {
  const script = ['f2f3', 'e7e5', 'g2g4', 'd8h4']
  let i = 0
  const scripted: Player = async () => ({ move: script[i++], pi: new Map([[script[i - 1], 1]]) })
  const g = await playGame(scripted, scripted, { maxPlies: 200, record: true })
  assert.equal(g.result, '0-1')
  assert.deepEqual(g.samples.map(s => s.z), [-1, 1, -1, 1])
  assert.equal(g.samples[0].pi.f2f3, 1)
})

test('the ply cap ends an unfinished game as a draw', async () => {
  const shuffle: Player = async c => ({ move: ['b1c3', 'c3b1', 'b8c6', 'c6b8'].find(u => c.moves({ verbose: true }).some(m => m.from + m.to === u))! })
  const g = await playGame(shuffle, shuffle, { maxPlies: 6, record: false })
  const replay = new Chess(); replay.loadPgn(g.pgn)
  assert.equal(replay.history().length, 6)
  assert.equal(g.plies, 6)
  assert.equal(g.result, '1/2-1/2')
})

test('buffer keeps the newest cap samples and round-trips disk', () => {
  const b = new ReplayBuffer()
  b.append(Array.from({ length: 30 }, (_, i) => ({ fen: String(i), pi: {}, z: 0 })), 20)
  assert.equal(b.samples.length, 20)
  assert.equal(b.samples[0].fen, '10')
  const path = mkdtempSync(tmpdir() + '/b') + '/buf.jsonl'
  b.save(path)
  assert.equal(ReplayBuffer.load(path).samples.length, 20)
})
