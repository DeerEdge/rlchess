import { test } from 'node:test'
import assert from 'node:assert/strict'
import { publish, type RunSummary, type GameRecord } from '../src/store.ts'
import { gunzipSync, gzipSync } from 'node:zlib'
import type { Sample } from '../src/game.ts'
import { playMany, score, trainingTier, rates } from '../src/worker.ts'
import { TIERS } from '../src/encode.ts'
import { EngineTimeout } from '../src/uci.ts'

const run: RunSummary = { ts: '2026-10-31T00:00:00Z', run: 72, tier: 'random', elo: 400, score40: 0.5, win40: 0.3, draw40: 0.4, plies: 80, buffer: 1000, seconds: 300, gamesTotal: 8, loss: 1.2, record: { random: { w: 1, d: 0, l: 0 } } }
const game: GameRecord = { ts: run.ts, tier: 'random', kind: 'eval', result: '1-0', pgn: '1. e4 *', agentWhite: true, plies: 1 }

const fakeKv = () => {
  const store = new Map<string, string>()
  return { store, get: async (k: string) => store.get(k) ?? null, put: async (k: string, v: string) => { store.set(k, v) } }
}

test('publish without CF_* env skips with a warning', async () => {
  delete process.env.CF_API_TOKEN
  const logs: string[] = [], warn = console.warn
  console.warn = (m: string) => { logs.push(m) }
  try { await publish(run, [game], { weights: '{}', buffer: '', state: '{}' }) } finally { console.warn = warn }
  assert.ok(logs.some(l => l.includes('publish skipped (no CF_API_TOKEN)')))
})

test('publish writes state, appends metrics (cap 20000), prepends games (cap 500), snapshots', async () => {
  const kv = fakeKv()
  kv.store.set('metrics', JSON.stringify(Array.from({ length: 20000 }, (_, i) => ({ run: i }))))
  kv.store.set('games', JSON.stringify(Array.from({ length: 500 }, (_, i) => ({ pgn: String(i) }))))
  await publish(run, [game], { weights: '{"w":1}', buffer: 'line\n', state: '{"run":96}', kv })
  assert.equal(JSON.parse(kv.store.get('state')!).run, 72)
  const metrics = JSON.parse(kv.store.get('metrics')!); assert.equal(metrics.length, 20000); assert.equal(metrics.at(-1).run, 72)
  const games = JSON.parse(kv.store.get('games')!); assert.equal(games.length, 500); assert.equal(games[0].pgn, '1. e4 *')
  assert.equal(kv.store.get('weights'), '{"w":1}'); assert.equal(gunzipSync(Buffer.from(kv.store.get('buffer')!, 'base64')).toString(), 'line\n')
  assert.equal(kv.store.get('worker-state'), '{"run":96}', 'the worker\'s own state is durable too, separate from the published summary')
  assert.equal(kv.store.get('weights/2026-10-31.json'), '{"w":1}', 'snapshot on every 72nd run')
})

test('playMany drops games whose engine times out instead of throwing', async () => {
  let n = 0
  const games = await playMany(4, async () => { if (n++ % 2) throw new EngineTimeout('x'); return { pgn: '', result: '1/2-1/2' as const, samples: [], agentWhite: true, plies: 0 } })
  assert.equal(games.length, 2)
})

test('playMany stops starting games once the deadline has passed', async () => {
  let started = 0
  const games = await playMany(10, async () => { started++; await new Promise(r => setTimeout(r, 30)); return { pgn: '', result: '1/2-1/2' as const, samples: [], agentWhite: true, plies: 0 } }, Date.now() + 50)
  assert.ok(started < 10 && games.length === started, `started ${started}`)
})

test('score is from the agent\'s colour, not the game index', () => {
  assert.equal(score({ result: '1-0', agentWhite: true }), 1)
  assert.equal(score({ result: '1-0', agentWhite: false }), 0)
  assert.equal(score({ result: '1/2-1/2', agentWhite: false }), 0.5)
})

test('restore pulls the worker files from kv and reports whether anything existed', async () => {
  const { restore } = await import('../src/store.ts')
  const { mkdtempSync, readFileSync } = await import('node:fs'); const { tmpdir } = await import('node:os')
  const dir = mkdtempSync(tmpdir() + '/r'), kv = fakeKv()
  assert.equal(await restore({ 'worker-state': dir + '/state.json' }, kv), false)
  kv.store.set('worker-state', '{"run":3}'); kv.store.set('buffer', gzipSync('a\nb\n').toString('base64'))
  assert.equal(await restore({ 'worker-state': dir + '/state.json', buffer: dir + '/buffer.jsonl' }, kv), true)
  assert.equal(readFileSync(dir + '/state.json', 'utf8'), '{"run":3}'); assert.equal(readFileSync(dir + '/buffer.jsonl', 'utf8'), 'a\nb\n', 'buffer is gunzipped')
})

test('training opponents are drawn from the tier below, the current tier and the tier above', () => {
  const seen = new Set(Array.from({ length: 200 }, () => trainingTier('stockfish-3').name))
  assert.deepEqual([...seen].sort(), ['stockfish-2', 'stockfish-3', 'stockfish-4'])
  assert.deepEqual([...new Set(Array.from({ length: 100 }, () => trainingTier(TIERS[0].name).name))].sort(), [TIERS[0].name, TIERS[1].name].sort(), 'bottom of the ladder has no tier below')
})

test('rates summarise the last 40 eval games of the current tier', () => {
  const h = [...Array(10).fill({ tier: 'x', score: 1 }), ...Array(20).fill({ tier: 'x', score: 0.5 }), ...Array(10).fill({ tier: 'x', score: 0 }), { tier: 'y', score: 1 }]
  assert.deepEqual(rates(h, 'x'), { score40: 0.5, win40: 0.25, draw40: 0.5 })
  assert.deepEqual(rates([], 'x'), { score40: 0.5, win40: 0, draw40: 0 })
})


test('annotate attaches one eval per position, start included, capped at the replay length', async () => {
  const { annotate, annotateSamples } = await import('../src/worker.ts')
  const g: GameRecord = { ...game, pgn: '1. e4 e5 2. Nf3 *', plies: 3 }
  await annotate([g], async fen => fen.split(' ')[1] === 'w' ? 10 : -10)
  assert.deepEqual(g.evals, [10, -10, 10, -10])
  const s: Sample[] = [{ fen: 'x w', pi: {}, z: 1 }, { fen: 'x b', pi: {}, z: -1 }]
  await annotateSamples(s, async () => 400) // +400 cp for white
  assert.deepEqual(s.map(x => x.v), [+Math.tanh(1).toFixed(3), -(+Math.tanh(1).toFixed(3))], 'from the side to move')
})

