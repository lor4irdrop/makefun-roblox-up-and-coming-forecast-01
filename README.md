# makefun-roblox-up-and-coming-forecast-01

Monte Carlo top-10 forecaster for Roblox's Up-and-Coming chart, for make.fun's
"Predict Roblox's Up-and-Coming top 10" bounty ($500 in USDC).

## Run it

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes `forecast.json`: one static probability distribution per place
(1–10) plus a `top10` guess, scored against every hourly chart read in the
72-hour window after entries close.

## Model

Each game's latent score (`-rank`) performs an hourly random walk for 72
hours. Score noise depends on the game's current rank band (calibrated from
the public chart archive: the top 10 moves slowly, deep ranks move fast).
Per-game drift is shrunk recent rank velocity (momentum) plus a small band
mean drift. Rare big jumps, chart exits, new-game entries and the 28-day
age-out (from `details.created` when present) are simulated explicitly.
Occupancy over the 72 hours, averaged over 1,200 simulations, gives
P(game at place k). The result is blended 25% with the house persistence
baseline. Python 3.13, standard library only, offline, deterministic
(fixed seed).

Local backtests (rolling 10-read windows on the practice archive, scorer
exactly per RULES): mean **-34.9** vs house baseline **-43.0**. For
reference, the house model forecast scores -37.2 on the official practice
and the best public entry -34.6 — so this entry is competitive but must
still beat the house on the final to take a prize.
