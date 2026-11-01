import { createRequire } from 'node:module'
import util from 'node:util'
;(util as any).isNullOrUndefined ??= (v: unknown) => v == null // tfjs-node 4.22 still calls this; Node ≥ 23 removed it
/** Native TensorFlow (CPU): real convolutions at a usable speed. Every module imports tf from here so the backend is registered once. */
const tf = createRequire(import.meta.url)('@tensorflow/tfjs-node') as typeof import('@tensorflow/tfjs-node')
export default tf
