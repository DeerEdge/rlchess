export type Record3 = { w: number; d: number; l: number }
export type RunSummary = { ts: string; run: number; tier: string; elo: number; score40: number; win40: number; draw40: number; plies: number; buffer: number; seconds: number; gamesTotal: number; loss: number; record: Record<string, Record3> }
export type GameRecord = { ts: string; tier: string; kind: 'tier' | 'selfplay' | 'eval'; result: string; pgn: string; agentWhite: boolean; plies: number; evals?: number[]; elo?: number; delta?: number }
/** What the worker is doing right now; the page polls this while a run is in flight. */
export type Live = { phase: 'play' | 'train' | 'eval'; run?: number; tier?: string; game?: number; games?: number; against?: string; agentWhite?: boolean; pgn?: string; batch?: number; batches?: number; loss?: number }
export type Kv = { get(key: string): Promise<string | null>; put(key: string, value: string): Promise<void> }

const METRICS_CAP = 20000, GAMES_CAP = 500, SNAPSHOT_EVERY = 96 // ≈ daily at a 15-minute cadence

/** Cloudflare KV over its REST API — no SDK. Needs CF_ACCOUNT_ID, CF_KV_NAMESPACE_ID, CF_API_TOKEN. */
export function cloudflareKv(): Kv | null {
  const { CF_ACCOUNT_ID: acct, CF_KV_NAMESPACE_ID: ns, CF_API_TOKEN: token } = process.env
  if (!token) return null
  const url = (key: string) => `https://api.cloudflare.com/client/v4/accounts/${acct}/storage/kv/namespaces/${ns}/values/${encodeURIComponent(key)}`
  const headers = { Authorization: `Bearer ${token}` }
  return {
    async get(key) { const r = await fetch(url(key), { headers }); return r.ok ? r.text() : null },
    async put(key, value) { const r = await fetch(url(key), { method: 'PUT', headers, body: value }); if (!r.ok) throw new Error(`kv put ${key}: ${r.status}`) },
  }
}

const json = async <T>(kv: Kv, key: string, fallback: T): Promise<T> => { const v = await kv.get(key); return v ? JSON.parse(v) : fallback }

/** Publish a run: summary/metrics/games for the page, the PGN archive, and the worker's live weights/buffer/state (durable). */
export async function publish(run: RunSummary, games: GameRecord[], files: { weights: string; buffer: string; state: string; kv?: Kv }) {
  const kv = files.kv ?? cloudflareKv()
  if (!kv) { console.warn('publish skipped (no CF_API_TOKEN)'); return }
  const metrics = await json<object[]>(kv, 'metrics', []), { record, ...point } = run
  const prev = await json<GameRecord[]>(kv, 'games', [])
  await kv.put('state', JSON.stringify(run))
  await kv.put('metrics', JSON.stringify([...metrics, point].slice(-METRICS_CAP)))
  await kv.put('games', JSON.stringify([...games, ...prev].slice(0, GAMES_CAP)))
  const day = run.ts.slice(0, 10)
  await kv.put(`archive/${day}/run-${run.run}.pgn`, games.map(g => g.pgn).join('\n\n'))
  await kv.put('weights', files.weights); await kv.put('buffer', files.buffer); await kv.put('worker-state', files.state)
  if (run.run % SNAPSHOT_EVERY === 0) await kv.put(`weights/${day}.json`, files.weights)
}

/** Throttled writer for the `live` key — KV writes are metered, so at most one per `everyMs`. Failures are ignored. */
export function liveReporter(kv: Kv | null = cloudflareKv(), everyMs = 60e3) {
  let last = -Infinity
  return async (payload: Live) => {
    if (!kv || Date.now() - last < everyMs) return
    last = Date.now()
    await kv.put('live', JSON.stringify({ ts: new Date().toISOString(), ...payload })).catch(e => console.warn(`live: ${e.message}`))
  }
}

/** Pull durable state from KV when the local files are missing (cache miss). Returns false if there is nothing yet. */
export async function restore(files: Record<string, string>, kv: Kv | null = cloudflareKv()): Promise<boolean> {
  if (!kv) return false
  const { writeFileSync } = await import('node:fs')
  let any = false
  for (const [key, path] of Object.entries(files)) { const v = await kv.get(key); if (v) { writeFileSync(path, v); any = true } }
  return any
}
