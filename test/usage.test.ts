import { test } from 'node:test'
import assert from 'node:assert/strict'
import { usage } from '../src/usage.ts'

const t0 = Date.parse('2026-11-03T06:22:05Z') / 1000
const vmstat = `procs -----------memory---------- ---swap-- -----io---- -system-- -------cpu------- -----timestamp-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st gu                 UTC
 1  0      0 15000000  10000 500000    0    0     0     0    0    0  2  1 97  0  0  0 2026-11-03 06:22:10
 4  0      0 12000000  10000 1000000    0    0     0     0    0    0 80 10 10  0  0  0 2026-11-03 06:22:15
 4  0      0 11000000  10000 1000000    0    0     0     0    0    0 90  5  5  0  0  0 2026-11-03 06:22:20`

test('usage turns vmstat samples and phase starts into the CPU timeline, averages and phase spans of a run', () => {
  const u = usage(vmstat, `play ${t0 + 5}\ntrain ${t0 + 10}\n`, t0, t0 + 20, 16_000_000)!
  assert.deepEqual(u.timeline, [[10, 90], [15, 95]], 'seconds since the start, busy % (100 − idle); the since-boot first line is skipped')
  assert.equal(u.cpu, 93)
  assert.equal(u.memPeakGB, 3.8, 'used = total − free − buffers − cache, peak, in GiB')
  assert.deepEqual(u.phases, [['setup', 0, 5], ['play', 5, 10], ['train', 10, 20]])
})

test('no samples, no usage (a local run has no vmstat log)', () => {
  assert.equal(usage('', '', t0, t0 + 20, 16_000_000), null)
})
