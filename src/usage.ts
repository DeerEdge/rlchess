export type Usage = { cpu: number; memPeakGB: number; phases: [string, number, number][]; timeline: [number, number][] }
export type Compute = Usage & { cores: number; memGB: number; model: string; inFlight: number }

/** A run's machine use from `vmstat -n -t 5` (UTC) and the workflow's "<phase> <epoch s>" lines: busy % (100 − idle) every 5 s, in seconds
 *  since `t0`; its mean; peak memory in use (total − free − buffers − cache, GiB); each phase's [start, end). Null without samples. */
export function usage(vmstat: string, phaseLog: string, t0: number, end: number, totalKB: number): Usage | null {
  const lines = vmstat.trim().split('\n'), head = lines.findIndex(l => / us +sy +id /.test(l))
  if (head < 0) return null
  const col = lines[head].trim().split(/\s+/), at = (k: string) => col.indexOf(k)
  const samples = lines.slice(head + 2).map(l => l.trim().split(/\s+/)).map(f => ({ // head + 1 is the since-boot average
    t: Date.parse(`${f.at(-2)}T${f.at(-1)}Z`) / 1000, busy: 100 - +f[at('id')], used: totalKB - +f[at('free')] - +f[at('buff')] - +f[at('cache')] }))
    .filter(s => [s.t, s.busy, s.used].every(Number.isFinite)) // a line in an unexpected format is dropped, never published as NaN
  if (!samples.length) return null
  const marks: [string, number][] = [['setup', 0], ...phaseLog.trim().split('\n').filter(Boolean).map(l => l.split(' ')).map(([n, s]): [string, number] => [n, +s - t0])]
  return {
    cpu: Math.round(samples.reduce((a, s) => a + s.busy, 0) / samples.length),
    memPeakGB: +(Math.max(...samples.map(s => s.used)) / 2 ** 20).toFixed(1),
    phases: marks.map(([n, s], i) => [n, s, i + 1 < marks.length ? marks[i + 1][1] : end - t0]),
    timeline: samples.map(s => [Math.round(s.t - t0), s.busy]),
  }
}
