import { createClient } from 'redis'
import { put } from '@vercel/blob'
import { readFile } from 'node:fs/promises'

export type Record3 = { w: number; d: number; l: number }
export type RunSummary = { ts: string; run: number; tier: string; elo: number; score40: number; gamesTotal: number; loss: number; record: Record<string, Record3> }
export type GameRecord = { ts: string; tier: string; kind: 'tier' | 'selfplay' | 'eval'; result: string; pgn: string }

type RedisLike = Pick<ReturnType<typeof createClient>, 'hSet' | 'lPush' | 'lTrim'>
const SNAPSHOT_EVERY = 96 // runs ≈ one day at a 15-minute cadence

/** Push a run's summary + games to Redis (and PGNs / a daily weights snapshot to Blob). No-ops without credentials. */
export async function publish(run: RunSummary, games: GameRecord[], deps: { redis?: RedisLike; weightsPath?: string } = {}) {
  let redis = deps.redis, own: ReturnType<typeof createClient> | undefined
  if (!redis) {
    if (!process.env.REDIS_URL) { console.warn('publish skipped (no REDIS_URL)'); return }
    own = createClient({ url: process.env.REDIS_URL }); await own.connect(); redis = own
  }
  const { record, ...flat } = run
  await redis.hSet('rlchess:state', { ...Object.fromEntries(Object.entries(flat).map(([k, v]) => [k, String(v)])), record: JSON.stringify(record) })
  await redis.lPush('rlchess:metrics', JSON.stringify({ ts: run.ts, run: run.run, tier: run.tier, elo: run.elo, score40: run.score40, gamesTotal: run.gamesTotal, loss: run.loss }))
  await redis.lTrim('rlchess:metrics', 0, 19999)
  if (games.length) { await redis.lPush('rlchess:games', games.map(g => JSON.stringify(g))); await redis.lTrim('rlchess:games', 0, 499) }
  await own?.quit()

  if (!process.env.BLOB_READ_WRITE_TOKEN) return
  const day = run.ts.slice(0, 10), opts = { access: 'public' as const, addRandomSuffix: false, allowOverwrite: true }
  await put(`rlchess/games/${day}/run-${run.run}.pgn`, games.map(g => g.pgn).join('\n\n'), opts)
  if (deps.weightsPath && run.run % SNAPSHOT_EVERY === 0) await put(`rlchess/weights/${day}.json`, await readFile(deps.weightsPath, 'utf8'), opts)
}
