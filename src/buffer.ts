import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import type { Sample } from './game.ts'

/** Fisher–Yates, in place. (`sort(() => Math.random() - 0.5)` is biased.) */
export function shuffle<T>(a: T[]): T[] {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] }
  return a
}

/** Replay buffer: JSON lines on disk, newest samples kept. */
export class ReplayBuffer {
  constructor(public samples: Sample[] = []) {}
  static load(path: string) {
    return new ReplayBuffer(existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [])
  }
  append(samples: Sample[], cap = 60000) { this.samples = [...this.samples, ...samples].slice(-cap) }
  save(path: string) { writeFileSync(path, this.samples.map(s => JSON.stringify(s)).join('\n') + '\n') }
}
