import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { Chess, type Square } from 'chess.js'
import tf from './tf.ts'
import { createNet, loadWeights, saveWeights, uciMoves } from './net.ts'
import { trainEpoch } from './train.ts'
import type { Sample } from './game.ts'
import { cloudflareKv, type Kv } from './store.ts'

type Pv = { cp?: number; mate?: number; line: string }
const CASTLE: Record<string, string> = { e1h1: 'e1g1', e1a1: 'e1c1', e8h8: 'e8g8', e8a8: 'e8c8' } // UCI_Chess960 writes castling as king-takes-rook
const LR = 1e-3, WEIGHTS = 'weights.json', STATE = 'state.json'

/** One line of Lichess's evaluation database (database.lichess.org, CC0) → a sample: the deepest eval's first move as a one-hot target over the
 *  legal moves, its score as the value target from the side to move (white's-view cp, mate = ±1000 like the loop's annotator). Null for an
 *  underpromotion, which the net cannot play. */
export function toSample(line: string): Sample | null {
  const { fen, evals } = JSON.parse(line) as { fen: string; evals: { depth: number; pvs: Pv[] }[] }
  const pv = evals.reduce((a, b) => (b.depth > a.depth ? b : a)).pvs[0], c = new Chess(`${fen} 0 1`)
  let best = pv.line.split(' ')[0]
  if (CASTLE[best] && c.get(best.slice(0, 2) as Square)?.type === 'k') best = CASTLE[best]
  const moves = uciMoves(c)
  if (!moves.includes(best)) return null
  const cp = pv.mate !== undefined ? Math.sign(pv.mate) * 1000 : Math.max(-1000, Math.min(1000, pv.cp!))
  return { fen: c.fen(), pi: Object.fromEntries(moves.map(m => [m, +(m === best)])), z: 0, v: +Math.tanh((c.turn() === 'w' ? cp : -cp) / 400).toFixed(3) }
}

const readLines = (f: string) => readFileSync(f, 'utf8').split('\n').filter(Boolean)

/** tsx src/pretrain.ts <chunk files of JSON lines>: one pass, one Adam across chunks, continuing weights.json if present; saved after every chunk,
 *  with its gradient steps added to the loop's step count in state.json, and progress written to kv `pretrain` for the page (done on the last). */
export async function pretrain(files: string[], kv: Kv | null = cloudflareKv()) {
  if (!files.length) throw new Error('no chunk files')
  const net = createNet(), opt = tf.train.adam(LR), t0 = Date.now(), total = files.reduce((n, f) => n + readLines(f).length, 0)
  if (existsSync(WEIGHTS)) await loadWeights(net, WEIGHTS)
  let seen = 0, skipped = 0, lines = 0
  for (const [i, f] of files.entries()) {
    const chunk = readLines(f); lines += chunk.length
    const samples = chunk.flatMap(l => { try { const s = toSample(l); if (s) return [s] } catch {} skipped++; return [] })
    const { policy, value, steps } = await trainEpoch(net, samples, { batch: 256, lr: LR, opt })
    await saveWeights(net, WEIGHTS); seen += samples.length
    if (existsSync(STATE)) { const s = JSON.parse(readFileSync(STATE, 'utf8')); s.steps = (s.steps ?? 0) + steps; writeFileSync(STATE, JSON.stringify(s, null, 1)) }
    await kv?.put('pretrain', JSON.stringify({ lines, total, positions: seen, done: i === files.length - 1, ts: new Date().toISOString() }))
    console.log(`pretrain: ${seen} positions (${skipped} skipped), policy ${policy.toFixed(4)}, value ${value.toFixed(4)}, ${Math.round((Date.now() - t0) / 1000)}s`)
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) pretrain(process.argv.slice(2)).catch(e => { console.error(e); process.exit(1) })
