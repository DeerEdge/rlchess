import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import { execSync } from 'node:child_process'
import { uciPlayer, EngineTimeout } from '../src/uci.ts'
import { TIERS } from '../src/encode.ts'
import { uciMoves } from '../src/net.ts'

const has = (bin: string) => { try { execSync(`which ${bin}`, { stdio: 'ignore' }); return true } catch { return false } }
const tier = (name: string) => TIERS.find(t => t.name === name)!

for (const [name, bin] of [['sf0-d1', 'stockfish'], ['sf0', 'stockfish'], ['sf-elo1320', 'stockfish']] as const) {
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

test('analyse gives centipawns from white\'s view (mate ±1000) and Stockfish\'s best move', { skip: !has('stockfish') }, async () => {
  const p = await uciPlayer(tier('sf20'), { movetime: 100, depth: 8 })
  try {
    const start = await p.analyse(new Chess().fen())
    assert.ok(Math.abs(start.cp) < 150, 'startpos is roughly level'); assert.ok(uciMoves(new Chess()).includes(start.best))
    assert.deepEqual(await p.analyse('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3'), { cp: -1000, best: '(none)' }, 'white is mated')
    assert.ok((await p.analyse('4k3/8/8/8/8/8/8/4K2Q w - - 0 1')).cp > 500, 'white up a queen, white to move')
    assert.ok((await p.analyse('4k3/8/8/8/8/8/8/4K2Q b - - 0 1')).cp > 500, 'white up a queen, black to move: still white\'s view')
  } finally { p.quit() }
})
