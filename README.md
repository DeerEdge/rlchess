# rlchess

A super tiny chess net (~86k params, dense MLP + MCTS) that learns in a loop that never stops, measured against a fixed roster of open-source engines. Progress is shown on [dheerajvislawath.com/blog/rlchess](https://dheerajvislawath.com/blog/rlchess).

## one run (`npm run run`)

1. load `state.json`, `weights.json`, `buffer.jsonl`
2. play `RLCHESS_GAMES` training games (half vs the current tier, half self-play, Dirichlet noise, `RLCHESS_SIMS` MCTS sims/move)
3. one training epoch over the replay buffer (last 20k positions)
4. `RLCHESS_EVAL` eval games vs the current tier — the **frozen protocol**: alternating colours, agent at full sims with temperature 1 for the first 4 plies then greedy (so the 10 games aren't the same game), engine at `movetime 100` / `depth 8`, 200-ply cap = draw
5. promote to the next tier when the last 40 eval games average ≥ 0.55; never demote
6. write the three files; publish state, metrics, games, the PGN archive and the live weights/buffer to Cloudflare KV (+ a daily weights snapshot)

GitHub Actions runs this every 15 minutes (`.github/workflows/loop.yml`). Training games stop starting after `RLCHESS_BUDGET_MS` (5 min) so a run fits the job. Hot state rides the Actions cache; durable state is KV (a cache miss restores from it). A tiny `state.json` commit every 96th run keeps GitHub from disabling the schedule after 60 idle days.

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
