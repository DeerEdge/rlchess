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

test('a slow engine times out and is dead afterwards, never answering a later position', async () => {
  const slow = await uciPlayer({ name: 'slow', rating: 0, cmd: ['node', '-e', 'process.stdin.on("data",d=>{if(/^uci/m.test(d))console.log("uciok");if(/isready/.test(d))console.log("readyok");if(/^go/m.test(d))setTimeout(()=>console.log("bestmove e2e4"),1500)})'] }, { movetime: 100, depth: 8, timeoutMs: 1000 })
  try {
    await assert.rejects(slow(new Chess()), EngineTimeout)
    const t = Date.now(); await assert.rejects(slow(new Chess()), EngineTimeout); assert.ok(Date.now() - t < 500, 'dead engine must reject immediately')
  } finally { slow.quit() }
})

test('a missing binary rejects instead of crashing', async () => {
  await assert.rejects(uciPlayer({ name: 'nope', rating: 0, cmd: ['definitely-not-a-real-binary-xyz'] }, { movetime: 100, depth: 8, timeoutMs: 1000 }), EngineTimeout)
})

test('a silent engine rejects with EngineTimeout', async () => {
  const p = await uciPlayer({ name: 'mute', rating: 0, cmd: ['node', '-e', 'process.stdin.on("data",d=>{if(/^uci/m.test(d))console.log("uciok");if(/isready/.test(d))console.log("readyok")})'] }, { movetime: 100, depth: 8, timeoutMs: 1000 })
  try { await assert.rejects(p(new Chess()), EngineTimeout) } finally { p.quit() }
})

test('an engine evaluates positions in centipawns from white\'s view, mate as ±1000', { skip: !has('stockfish') }, async () => {
  const p = await uciPlayer(tier('stockfish-20'), { movetime: 100, depth: 8 })
  try {
    assert.ok(Math.abs(await p.evaluate(new Chess().fen())) < 150, 'startpos is roughly level')
    assert.equal(await p.evaluate('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3'), -1000, 'white is mated')
    assert.ok(await p.evaluate('4k3/8/8/8/8/8/8/4K2Q w - - 0 1') > 500, 'white up a queen, white to move')
    assert.ok(await p.evaluate('4k3/8/8/8/8/8/8/4K2Q b - - 0 1') > 500, 'white up a queen, black to move: still white\'s view')
  } finally { p.quit() }
})
