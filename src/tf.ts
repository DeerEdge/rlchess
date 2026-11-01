import { createRequire } from 'node:module'
import type * as T from '@tensorflow/tfjs-node'
import util from 'node:util'
;(util as any).isNullOrUndefined ??= (v: unknown) => v == null // tfjs-node 4.22 still calls this; Node ≥ 23 removed it
/** Native TensorFlow (CPU): real convolutions at a usable speed. Every module imports tf from here so the backend is registered once. */
const tf = createRequire(import.meta.url)('@tensorflow/tfjs-node') as typeof T
// type-only twin so `tf.LayersModel` etc. work as types, as with a namespace import
declare namespace tf { type LayersModel = T.LayersModel; type Tensor = T.Tensor; type Scalar = T.Scalar; type SymbolicTensor = T.SymbolicTensor }
export default tf
