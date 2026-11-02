import type { Chess } from 'chess.js'
import { readFileSync } from 'node:fs'

export const PLANES = 18 // 12 pieces, side-to-move, 4 castling, en-passant
/** A ladder rung: in-page mover (no cmd) or UCI engine. `depth` overrides the protocol's depth 8; `random` is the share of moves played at random
 *  instead (an Elo World–style dilution); `anchor` is the rung whose rating is fixed (Stockfish's own UCI_Elo calibration). */
export type Tier = { name: string; rating: number; cmd?: string[]; options?: Record<string, string>; depth?: number; random?: number; anchor?: boolean }
export const TIERS: Tier[] = JSON.parse(readFileSync(new URL('../tiers.json', import.meta.url), 'utf8'))

const PIECE = 'pnbrqk'
const sq = (file: number, rank: number, flip: boolean) => (flip ? 7 - rank : rank) * 8 + file

/** Square index 0-63 (a1 = 0) of a uci square, mirrored by rank when `flip`. */
export const squareIndex = (s: string, flip = false) => sq(s.charCodeAt(0) - 97, +s[1] - 1, flip)
export const moveIndex = (uci: string, flip = false) => ({ from: squareIndex(uci, flip), to: squareIndex(uci.slice(2), flip) })

/** Board → 18×64 planes from the side-to-move's view: planes 0-5 my pieces, 6-11 theirs. */
export function encode(chess: Chess): Float32Array {
  const me = chess.turn(), flip = me === 'b', x = new Float32Array(PLANES * 64)
  chess.board().forEach((row, r) => row.forEach((p, f) => {
    if (p) x[((p.color === me ? 0 : 6) + PIECE.indexOf(p.type)) * 64 + sq(f, 7 - r, flip)] = 1
  }))
  x.fill(1, 12 * 64, 13 * 64)
  const c = chess.getCastlingRights(me), o = chess.getCastlingRights(me === 'w' ? 'b' : 'w')
  ;[c.k, c.q, o.k, o.q].forEach((on, i) => on && x.fill(1, (13 + i) * 64, (14 + i) * 64))
  const ep = chess.fen().split(' ')[3]
  if (ep !== '-') x[17 * 64 + squareIndex(ep, flip)] = 1
  return x
}
