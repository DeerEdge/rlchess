import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import type { Sample } from './game.ts'

/** Replay buffer: JSON lines on disk, newest samples kept. */
export class ReplayBuffer {
  constructor(public samples: Sample[] = []) {}
  static load(path: string) {
    return new ReplayBuffer(existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [])
  }
  append(samples: Sample[], cap = 20000) { this.samples = [...this.samples, ...samples].slice(-cap) }
  save(path: string) { writeFileSync(path, this.samples.map(s => JSON.stringify(s)).join('\n') + '\n') }
}
