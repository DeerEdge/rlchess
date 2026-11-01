import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { execSync } from 'node:child_process'
import { uciPlayer, EngineTimeout } from '../src/uci.ts'
import { TIERS } from '../src/encode.ts'
import { uciMoves } from '../src/net.ts'

const has = (bin: string) => { try { execSync(`which ${bin}`, { stdio: 'ignore' }); return true } catch { return false } }
const tier = (name: string) => TIERS.find(t => t.name === name)!

for (const [name, bin] of [['lozza', 'node'], ['sunfish', 'python3'], ['stockfish-0', 'stockfish']] as const) {
  test(`${name} answers a legal bestmove on startpos`, { skip: !has(bin) }, async () => {
    const p = await uciPlayer(tier(name), { movetime: 100, depth: 8 })
    try {
      const chess = new Chess()
      const { move } = await p(chess)
      assert.ok(uciMoves(chess).includes(move), `${move} not legal`)
    } finally { p.quit() }
  })
}

test('a silent engine rejects with EngineTimeout', async () => {
  const p = await uciPlayer({ name: 'mute', rating: 0, cmd: ['node', '-e', 'process.stdin.on("data",d=>{if(/^uci/m.test(d))console.log("uciok");if(/isready/.test(d))console.log("readyok")})'] }, { movetime: 100, depth: 8, timeoutMs: 1000 })
  try { await assert.rejects(p(new Chess()), EngineTimeout) } finally { p.quit() }
})
