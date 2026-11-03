import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

test('no npm script is named pre<another script>: npm would run it as a hook before that script', () => {
  const scripts = Object.keys(JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')).scripts)
  assert.deepEqual(scripts.filter(s => scripts.includes(s.replace(/^(pre|post)/, '')) && /^(pre|post)/.test(s)), [])
})
