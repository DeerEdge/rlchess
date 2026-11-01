# rlchess — design

Date: 2026-10-31. Status: builder-checked; awaiting owner approval.

## Intent

A **super tiny** chess agent that improves by RL in a loop that never stops, playing against fixed open-source opponents (Stockfish at skill tiers plus a roster of small open-source engines), with its progress shown publicly at `/blog/rlchess` on the portfolio. Visitors watch; they don't interact.

This is phase one of a larger idea: a Polytopia-style pixel world with the same loop (worker → DB → public dashboard), learning agent vs fixed bots and hazards. Chess goes first because the rules engine, the opponents and the metric (Elo) all exist already, so the only thing being built is the loop. Phase two swaps the engine and the renderer; everything else carries over. See `experiments/sim-world/RESEARCH.md`.

Influences: Prime Intellect's *General Agent* (difficulty only steps up once the measured success rate is in a target band) and *On the Nature of the Swarm* (persistent state across runs, not one-shot). Visual language: the pixel-lab halftone look already on the site.

## Goals

- A line that goes up: Elo (estimated vs Stockfish tiers) over time, visible to anyone.
- Continuous: the loop runs unattended for months; the page always shows the latest.
- Cheap: steps can be slow and small. Target ≤ $0 fixed cost, inference-free (no LLM calls — pure RL).
- Honest: a measured curriculum. The agent moves to a harder tier only when it beats the current one at ≥55% over the last 40 eval games.
- Replayable: every game is stored; the page can scrub through any past game.

## Non-goals

- Beating strong Stockfish. Plateauing around skill level 3–6 is expected and fine.
- Visitor interaction (playing it, uploading agents, voting).
- Opening books, endgame tablebases, or hand-written chess heuristics in the learner.
- A general RL framework. One learner, one game.

## Anti-goals (would make it a failure even if it "works")

- A curve that goes up because the opponent got weaker or the eval changed — the eval protocol is frozen from day one.
- Costs that creep (always-on GPU, paid inference). If it needs money to keep running, it's wrong.
- The page showing stale data without saying so. Staleness must be visible ("last step 3h ago").

## Constraints

- Site is React/Vite on Vercel. Vercel hobby cron is once a day — unusable for the loop. Serverless functions time out in 10–60 s.
- Existing infra to reuse: Redis (`REDIS_URL`, `api/visitors.js` pattern) and `@vercel/blob` (already a dependency).
- The learner must run on CPU. Small net, few MCTS simulations.
- Public page is `/blog/rlchess`: a tab in the blog's left rail, same 3-column layout, dark theme, mono labels, halftone aesthetic.

## Decisions (made)

### Where the loop runs
**GitHub Actions scheduled workflow**, every 15 minutes. Each run: checkout (weights + buffer) → K training games (half vs current tier, half self-play) → one training epoch → 10 eval games → commit weights/buffer, push metrics + games to Redis, archive PGNs. Target ≤ 10 minutes per run on a 2-vCPU runner; minutes are unlimited on a public repo, so the cap is about keeping runs frequent, not cost. K is whatever fits (likely 6–12 games at 30–60 sims/move). "Slow and small" is the point — ~100 runs/day, each a nudge. A `concurrency` group guards against overlapping runs.

The portfolio repo is **private**, and private repos get 2,000 free Actions minutes/month — that cadence would burn ~17,000. So the loop lives in a **separate public repo** (`DeerEdge/rlchess`), where Actions minutes are unlimited; it holds `engine/`, `worker.ts` and the workflow, and talks to Blob/Redis with secrets. The portfolio repo keeps only the page and the API route. (Alternative if a second repo is unwanted: same repo, every 2 hours, ~1,500 min/month — fits, but 12 runs/day makes the curve crawl.)
*Fallback if Actions proves too flaky:* a Render background worker running the same script as a `while(true)` with a sleep (~$7/mo).

### Storage
- **Weights + replay buffer:** live in the `rlchess` repo itself. Each run commits the updated `weights.bin` and `buffer.bin` back to the repo (one bot commit per run, squashed weekly if history bloats). Reasons: (a) no per-run Blob transfer — a 1M-param net plus a 20k-position buffer round-tripped 100×/day is tens of GB/month, which blows past Vercel Hobby's included Blob bandwidth and violates the cost anti-goal; (b) GitHub disables scheduled workflows on public repos after 60 days with no commits, and the bot commits keep the loop alive by construction. Vercel Blob is used only for a **dated weights snapshot once a day** (rollback) and the **PGN archive** (`rlchess/games/YYYY-MM-DD.pgn`).
- **Metrics + game log:** Redis. `rlchess:metrics` — a list of `{ts, run, tier, elo, winrate40, gamesTotal, loss}` entries appended per run (capped at 20k). `rlchess:games` — a capped list of the last 500 games as `{ts, tier, result, pgn, movesWithPolicy?}`; the full PGN archive also goes to Blob as `rlchess/games/YYYY-MM-DD.pgn`.
- **Current state:** `rlchess:state` — one JSON hash the page polls: tier, elo, last step time, current/most-recent game id.
No Postgres. If the metrics list ever matters enough to query, migrate then.

### Learner
- Rules: `chess.js`.
- **Opponents — a fixed roster, not just Stockfish.** Every opponent is a tier with its own estimated rating in `tiers.json`:
  - tier −1: uniform random mover (400)
  - open-source engines that run in Node or as a subprocess, roughly in strength order: **p4wn** (JS, ~1100), **Sunfish** (Python, ~1500 at low depth; needs `python3` on the runner), **Lozza** or **Garbochess-JS** (JS UCI, ~1800–2200 at shallow depth), then **Stockfish** skill 0 … 20 (800 → 2800) to extend the ladder past them. Exact set and ordering are fixed before the first run and never silently changed; a new engine added later is a new row on the chart, not a reshuffle.
  - Each engine runs at a fixed, documented time/depth cap so it's deterministic and cheap.
- Promotion walks the roster in rating order. The dashboard shows a **record vs every opponent** (W/D/L), not only the current tier — "beats Sunfish, 30% vs Lozza" is the headline people will read.
- Net: policy + value, **tiny by design** — the smallness is part of the story ("this is 60 KB of weights and it beats X"). Board → 8×8×12 planes → 2–3 small conv layers (16–32 filters) → policy logits over a fixed 1858-move encoding + scalar value. Budget: **≤100k params** (~400 KB fp32, ~100 KB fp16), so a forward pass is sub-millisecond on CPU and weights fit in a git commit without thinking. If 100k plateaus embarrassingly early, the doc allows one step up to 300k; anything bigger is a different project.
- Self-play with MCTS, ~30–60 simulations/move, Dirichlet noise at root, temperature 1 for the first 20 plies then greedy.
- Training: replay buffer of the last ~20k positions (in Blob), one epoch of minibatch SGD per run.
- Curriculum **shapes training, not just measurement**: each run plays half its training games against the current tier (as the opponent, both colours) and half as self-play. Tier games are a much stronger early signal than self-play from a random net; self-play keeps the policy from overfitting to Stockfish's style. Games are tagged `kind: tier | selfplay | eval`; only `eval` games count toward promotion.
- Tiers: −1 = uniform random mover, then Stockfish skill 0, 1, 2 … 20. Promote at ≥55% score over the **rolling last 40 eval games**; never demote (demotion would make the curve lie).
- **Frozen eval protocol** (changing any of this later = a new series on the chart, never a silent change): 10 eval games per run, alternating colours, agent at its full self-play search budget (same sims/move), Stockfish at the tier's skill with `movetime 100ms` and depth cap 8, max 200 plies then adjudicated a draw, threefold/50-move/insufficient-material per `chess.js`.
- **Rating table** (fixed in the repo as `tiers.json`, with the source noted): tier −1 = 400; Stockfish skill 0 → 20 mapped linearly 800 → 2800 at the depth cap above. These are community estimates, not official; the page shows raw score vs tier alongside Elo so the number is never the only evidence.
- Elo estimate: standard formula from the rolling score vs the tier rating, clamped to avoid ±∞ at 0% / 100%.

### Public page `/blog/rlchess`
- Route is `/blog/rlchess` on the portfolio domain (on `dheerajvislawath.blog` it is `/rlchess` via the existing host rewrite — the page must use absolute `/api/rlchess` so polling works from either). The blog shell already exists at `src/blog/Blog.jsx`: dark theme (the `text-gray-*` classes are remapped to light text by `src/index.css`), 3-column grid, left rail with `← home`; it has no post tabs yet. Add `rlchess` to that rail under `← home` and render the page in the content column. Nothing else about the blog changes.
- Content column: live/latest board rendered to canvas as a pixel board (halftone pass on top, green accent), latest game auto-replays move by move; a scrubber; below it the dashboard: Elo over time (line), current tier + win rate over last 40, total games, total runs, "last step N min ago" (red if > 2h), and a small table of the last 10 games (tier, result, length).
- Data comes from a new `api/rlchess.js` serverless route that reads Redis (readonly) and returns JSON; the page polls every 60 s.
- No framework additions beyond what's installed; charts are inline SVG.

### Repo layout
```
DeerEdge/rlchess (public)          DeerEdge/portfolio (this repo)
  engine/   # net, mcts, selfplay,    api/rlchess.js        # readonly Redis → JSON
            # train, eval — plain TS  src/blog/RlChess.jsx  # /blog/rlchess page
  worker.ts # one run: load → play →  experiments/sim-world/RESEARCH.md
            # train → eval → save
  .github/workflows/loop.yml
  README.md
```

## Left to the builder

- Exact net architecture and hyperparameters, as long as a run fits in the Actions budget.
- Move encoding details; whether to use `chess.js` legality masking in the policy head (recommended) or filter after.
- MCTS implementation details, batching, reuse of the search tree between moves.
- How Stockfish is invoked in Actions (WASM via npm vs apt-installed binary) — whichever is simpler and deterministic.
- Board rendering style within the constraints (pixel pieces, halftone pass).
- How the replay scrubber and polling are structured in React.

## Phase two (not in scope, recorded so phase one doesn't block it)

Same worker/Blob/Redis/API/page pattern with the chess engine replaced by a tiny seeded Polytopia-like sim and the opponents replaced by scripted factions + hazards. The dashboard, the curriculum logic, and the replay machinery should be written so that "game" is an interface, not chess-specific, where it costs nothing to do so — but not at the expense of shipping chess.


## Amendments (2026-10-31, during build)

- **Storage is Cloudflare KV** (owner decision), not Redis + Vercel Blob. Keys: `state`, `metrics`, `games`, `archive/<day>/run-<n>.pgn`, `weights`, `buffer`, `weights/<day>.json`. Env: `CF_ACCOUNT_ID`, `CF_KV_NAMESPACE_ID`, `CF_API_TOKEN`. Durable state is KV; the Actions cache is the fast path; a `state.json` commit every 96th run only keeps the schedule alive.
- **Net is a dense MLP** (1152→64→64→128+1, 86k params): tfjs's pure-JS conv backprop is ~40× too slow on CPU.
- **Ladder is sorted by rating**: random 400, stockfish-0…7 800–1500, sunfish 1500, stockfish-8…12, lozza 2000, stockfish-13…20. p4wn dropped (not a UCI engine).
- **Eval protocol**: agent uses temperature 1 for the first 4 plies, then greedy — otherwise the 10 eval games against a deterministic engine collapse into ~2 distinct games.
- **Run budget**: training games stop starting after 5 min; training uses a 4096-sample subset of the buffer per run.
- **Page route is `/rlchess`** (a section beside `blog` in the rails), not under the blog.
