import { test } from 'node:test'
import assert from 'node:assert/strict'
import { publish, type RunSummary, type GameRecord } from '../src/store.ts'
import { playMany } from '../src/worker.ts'
import { EngineTimeout } from '../src/uci.ts'

const run: RunSummary = { ts: '2026-10-31T00:00:00Z', run: 1, tier: 'random', elo: 400, score40: 0.5, gamesTotal: 8, loss: 1.2, record: { random: { w: 1, d: 0, l: 0 } } }
const game: GameRecord = { ts: run.ts, tier: 'random', kind: 'eval', result: '1-0', pgn: '1. e4 *' }

test('publish without REDIS_URL skips with a warning', async () => {
  delete process.env.REDIS_URL
  const logs: string[] = [], warn = console.warn
  console.warn = (m: string) => { logs.push(m) }
  try { await publish(run, [game]) } finally { console.warn = warn }
  assert.ok(logs.some(l => l.includes('publish skipped (no REDIS_URL)')))
})

test('publish writes state, metrics and games with trims', async () => {
  const calls: string[] = []
  const redis = {
    hSet: async (k: string) => { calls.push(`hSet ${k}`) },
    lPush: async (k: string) => { calls.push(`lPush ${k}`) },
    lTrim: async (k: string, a: number, b: number) => { calls.push(`lTrim ${k} ${a} ${b}`) },
  }
  await publish(run, [game], { redis: redis as never })
  assert.deepEqual(calls, ['hSet rlchess:state', 'lPush rlchess:metrics', 'lTrim rlchess:metrics 0 19999', 'lPush rlchess:games', 'lTrim rlchess:games 0 499'])
})

test('playMany drops games whose engine times out instead of throwing', async () => {
  let n = 0
  const games = await playMany(4, async () => { if (n++ % 2) throw new EngineTimeout('x'); return { pgn: '', result: '1/2-1/2', samples: [] } })
  assert.equal(games.length, 2)
})
