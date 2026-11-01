import { Chess } from 'chess.js'
import type * as tf from '@tensorflow/tfjs'
import { predict, legalPolicy, uciMoves } from './net.ts'

type Node = { n: number; w: number; p: number; children?: Map<string, Node> }
const C_PUCT = 1.5

const apply = (c: Chess, uci: string) => c.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] })

async function expand(net: tf.LayersModel, node: Node, c: Chess) {
  const moves = uciMoves(c)
  const { from, to, value } = await predict(net, c)
  const pi = legalPolicy(from, to, moves, c.turn() === 'b')
  node.children = new Map(moves.map(m => [m, { n: 0, w: 0, p: pi.get(m)! }]))
  return value
}

// Marsaglia-Tsang gamma sampler; Dirichlet(α) = normalised gammas.
function gamma(a: number): number {
  if (a < 1) return gamma(a + 1) * Math.pow(Math.random(), 1 / a)
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d)
  for (;;) {
    let x: number, v: number
    do { x = normal(); v = 1 + c * x } while (v <= 0)
    v *= v * v
    if (Math.log(Math.random()) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v
  }
}
const normal = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())

function addNoise(root: Node, alpha = 0.3, eps = 0.25) {
  const g = [...root.children!.values()].map(() => gamma(alpha)), z = g.reduce((a, b) => a + b, 0)
  ;[...root.children!.values()].forEach((ch, i) => { ch.p = (1 - eps) * ch.p + eps * g[i] / z })
}

/** PUCT search from `chess`. Returns the chosen move (null if none) and the visit distribution. */
export async function search(net: tf.LayersModel, chess: Chess, sims: number, opts: { noise?: boolean; temperature?: number } = {}) {
  const root: Node = { n: 0, w: 0, p: 1 }
  if (!uciMoves(chess).length) return { move: null, pi: new Map<string, number>() }
  await expand(net, root, chess)
  if (opts.noise) addNoise(root)
  for (let i = 0; i < sims; i++) {
    const c = new Chess(chess.fen()), path = [root]
    let node = root
    while (node.children) {
      let best: [string, Node] | undefined, bestScore = -Infinity
      for (const [m, ch] of node.children) {
        const s = (ch.n ? ch.w / ch.n : 0) + C_PUCT * ch.p * Math.sqrt(node.n) / (1 + ch.n)
        if (s > bestScore) { bestScore = s; best = [m, ch] }
      }
      apply(c, best![0]); node = best![1]; path.push(node)
    }
    // value from the perspective of the side to move at the leaf
    let v = c.isGameOver() ? (c.isCheckmate() ? -1 : 0) : await expand(net, node, c)
    for (let k = path.length - 1, sign = -1; k >= 0; k--, sign = -sign) { path[k].n++; path[k].w += sign * v }
  }
  const T = opts.temperature ?? 1, entries = [...root.children!].map(([m, ch]) => [m, ch.n] as const)
  const total = entries.reduce((s, [, n]) => s + n, 0)
  const pi = new Map(entries.map(([m, n]) => [m, n / total]))
  if (T === 0) return { move: entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0], pi }
  const weights = entries.map(([, n]) => Math.pow(n, 1 / T)), wsum = weights.reduce((a, b) => a + b, 0)
  let r = Math.random() * wsum, move = entries[entries.length - 1][0]
  for (let i = 0; i < entries.length; i++) if ((r -= weights[i]) <= 0) { move = entries[i][0]; break }
  return { move, pi }
}
