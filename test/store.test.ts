import { test } from 'node:test'
import assert from 'node:assert/strict'
import { publish, type RunSummary, type GameRecord } from '../src/store.ts'
import { playMany, score } from '../src/worker.ts'
import { EngineTimeout } from '../src/uci.ts'

const run: RunSummary = { ts: '2026-10-31T00:00:00Z', run: 96, tier: 'random', elo: 400, score40: 0.5, gamesTotal: 8, loss: 1.2, record: { random: { w: 1, d: 0, l: 0 } } }
const game: GameRecord = { ts: run.ts, tier: 'random', kind: 'eval', result: '1-0', pgn: '1. e4 *' }

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

test('publish writes state, appends metrics (cap 20000), prepends games (cap 500), archives, snapshots', async () => {
  const kv = fakeKv()
  kv.store.set('metrics', JSON.stringify(Array.from({ length: 20000 }, (_, i) => ({ run: i }))))
  kv.store.set('games', JSON.stringify(Array.from({ length: 500 }, (_, i) => ({ pgn: String(i) }))))
  await publish(run, [game], { weights: '{"w":1}', buffer: 'line\n', state: '{"run":96}', kv })
  assert.equal(JSON.parse(kv.store.get('state')!).run, 96)
  const metrics = JSON.parse(kv.store.get('metrics')!); assert.equal(metrics.length, 20000); assert.equal(metrics.at(-1).run, 96)
  const games = JSON.parse(kv.store.get('games')!); assert.equal(games.length, 500); assert.equal(games[0].pgn, '1. e4 *')
  assert.equal(kv.store.get('archive/2026-10-31/run-96.pgn'), '1. e4 *')
  assert.equal(kv.store.get('weights'), '{"w":1}'); assert.equal(kv.store.get('buffer'), 'line\n')
  assert.equal(kv.store.get('worker-state'), '{"run":96}', 'the worker\'s own state is durable too, separate from the published summary')
  assert.equal(kv.store.get('weights/2026-10-31.json'), '{"w":1}', 'snapshot on every 96th run')
})

test('playMany drops games whose engine times out instead of throwing', async () => {
  let n = 0
  const games = await playMany(4, async () => { if (n++ % 2) throw new EngineTimeout('x'); return { pgn: '', result: '1/2-1/2', samples: [], agentWhite: true } })
  assert.equal(games.length, 2)
})

test('playMany stops starting games once the deadline has passed', async () => {
  let started = 0
  const games = await playMany(10, async () => { started++; await new Promise(r => setTimeout(r, 30)); return { pgn: '', result: '1/2-1/2', samples: [], agentWhite: true } }, Date.now() + 50)
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
  kv.store.set('worker-state', '{"run":3}')
  assert.equal(await restore({ 'worker-state': dir + '/state.json' }, kv), true)
  assert.equal(readFileSync(dir + '/state.json', 'utf8'), '{"run":3}')
})
