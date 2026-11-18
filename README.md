# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/51prass/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/51prass/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), from hour 1's print only. For every hour 2–24 it writes five quantiles (5/25/50/75/95%) of that hour's concurrent viewers. The model is a per-hour ridge regression of log(viewers_h / viewers_hour1) on three features — log hour-1 viewers, log hour-1 channels, and the log median pre-release level — fitted over past big game launches in the case's comparables, with the spread taken from that hour's residual standard deviation. A rolling-midnight launch does not stay flat: hour 1 holds only the regions already unlocked, and the climb through the day depends on how big hour 1 already is. Leave-one-out over the kit's past launches: mean pinball loss ~0.25 on the launches most like GTA VI, against ~0.40 for the persistence baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
