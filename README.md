# rlchess

A small chess net (142k params, conv tower + MCTS) that learns in a loop that never stops, measured against a fixed ladder of engines. Progress is shown on [dheerajvislawath.com/rlchess](https://dheerajvislawath.com/rlchess).

## one run (`.github/workflows/loop.yml`, every ~20 minutes; each run dispatches the next)

`src/worker.ts` has one mode per step; four shards run in parallel, one per runner core.

1. **restore** — on an Actions-cache miss, pull state, weights and the replay buffer from Cloudflare KV
2. **play** ×4, then **train** — three times over:
   - play: games for ~3.5 min, half vs the training pool, half self-play, with Dirichlet noise and `RLCHESS_SIMS` simulations per move
   - train: merge the shards into the 60k replay buffer, one Adam pass (lr 1e-3, batch 256) over 32k samples, loss `−Σ π log p + (v* − v)²`
3. **eval** ×4 — `RLCHESS_EVAL` (24) rated games, 6 against each of the 4 ladder rungs rated nearest the agent (the panel rotates as the rating moves), under the **frozen protocol**: alternating colours, temperature 1 for 4 plies then greedy, engines at `movetime 100` / `depth 8`, 200-ply cap = draw; each game gets per-ply Stockfish evals
4. **publish** — standard Elo (K = 32) per rated game against that opponent's calibrated rating; average centipawn loss of the agent's moves; separate policy and value losses; write state, metrics, games, weights and buffer to KV

Training policy targets are the search's visit counts (or an opponent engine's move), blended half-and-half with Stockfish depth 8's best move; value targets are Stockfish's eval, `tanh(cp/400)`. The training pool is the 6 rungs rated nearest the agent, plus self-play.

## the ladder (`tiers.json`, measured by `npm run calibrate` / `.github/workflows/calibrate.yml`)

Random and greedy movers, Stockfish skill 0 diluted with 75/50/25/10% random moves (as in Elo World), skill 0 at depth 1/5/8, skill 3–20, and Stockfish's `UCI_Elo 1320` as the anchor. Every rung plays its next two and the anchor plays every rung, 40 games each, on the loop's runner and Stockfish build; a maximum-likelihood Bradley–Terry fit with the anchor fixed gives the ratings (`calibration.json` has the raw results). Ratings are on Stockfish's CCRL-anchored scale, not human Elo.

Hot state rides the Actions cache; durable state is KV (6 writes per run, inside the free plan).

## local

```
npm i && brew install stockfish
npm test
RLCHESS_GAMES=2 RLCHESS_SIMS=10 RLCHESS_EVAL=4 RLCHESS_BUDGET_MS=20000 npm run run   # ~1 min, no publish without CF_API_TOKEN
```

Secrets for the workflow: `CF_ACCOUNT_ID`, `CF_KV_NAMESPACE_ID`, `CF_API_TOKEN` (Workers KV Storage: Edit).

## rules that keep the curve honest

- Ladder ratings change only by re-running the calibration; `calibration.json` keeps the games behind them.
- The eval protocol above never changes silently; a change is a new series on the chart.
- The agent's Elo is on the ladder's scale: Stockfish's UCI_Elo 1320 is the only fixed point, everything else is measured.

See `docs/design.md` for the original design.
