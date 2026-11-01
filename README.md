# rlchess

A small chess net (142k params, conv tower + MCTS) that learns in a loop that never stops, measured against a fixed ladder of engines. Progress is shown on [dheerajvislawath.com/rlchess](https://dheerajvislawath.com/rlchess).

## one run (`.github/workflows/loop.yml`, every ~20 minutes; each run dispatches the next)

`src/worker.ts` has one mode per step; four shards run in parallel, one per runner core.

1. **restore** — on an Actions-cache miss, pull state, weights and the replay buffer from Cloudflare KV
2. **play** ×4, then **train** — three times over:
   - play: games for ~3.5 min, half vs a pool (the rung below, the current rung, the rung above, Stockfish skill 3/8/14/20 as teachers), half self-play, with Dirichlet noise and `RLCHESS_SIMS` simulations per move. Policy targets are the search's visit counts for the agent's moves and the engine's move (one-hot over legal moves) for engine moves; value targets are Stockfish depth-8 evals, `tanh(cp/400)`
   - train: merge the shards into the 60k replay buffer, one Adam pass (lr 1e-3, batch 256) over 32k samples, loss `−Σ π log p + (v* − v)²`
3. **eval** ×4 — `RLCHESS_EVAL` (24) rated games vs the current rung, the **frozen protocol**: alternating colours, temperature 1 for 4 plies then greedy, engines at `movetime 100` / `depth 8`, 200-ply cap = draw; each game gets per-ply Stockfish evals
4. **publish** — standard Elo (K = 32) per rated game against the rung's fixed rating; promote when the last 40 rated games score ≥ 0.55, never demote; average centipawn loss of the agent's moves as a direct strength signal; write state, metrics, games, weights and buffer to KV

Hot state rides the Actions cache; durable state is KV (6 writes per run, inside the free plan).

## local

```
npm i && brew install stockfish   # python3 and node are the other two opponents
npm test
RLCHESS_GAMES=2 RLCHESS_SIMS=10 RLCHESS_EVAL=2 npm run run   # ~1 min, no publish without CF_API_TOKEN
```

Secrets for the workflow: `CF_ACCOUNT_ID`, `CF_KV_NAMESPACE_ID`, `CF_API_TOKEN` (Workers KV Storage: Edit).

## rules that keep the curve honest

- `tiers.json` is append-only. Never reorder or re-rate a row once the loop has run.
- The eval protocol above never changes silently; a change is a new series on the chart.
- Elo is an estimate from the score vs the tier's labelled rating (community estimates). The raw score is always shown next to it.

See `docs/design.md` for the full design and `engines/README.md` for the vendored opponents.
