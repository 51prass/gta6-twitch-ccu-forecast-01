"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Model: for each hour h = 2..24, the log-ratio ln(viewers_h / viewers_hour1) is
modelled as a linear function of three features of the case at hand, fitted by
ridge regression over past big game launches (the case's `comparables`):

    ln(v_h / v_1) ~ b0 + b1 * ln(v_1) + b2 * ln(channels_1) + b3 * ln(pre24)

where v_1 is hour 1's print, channels_1 its live channels, and pre24 the median
hourly viewers over the 24 hours before the release day (the game's own
pre-release level). The fit is done per hour, over the comparables that look
like a rolling-midnight console launch with a real hour 1: a complete release
day, hour-1 viewers >= 5000, and not an early-access-contaminated, weak or
non-launch row (the kit's own "Tier:" annotation in each comparable's unlock
note). The forecast's median is p1 * exp(fitted log-ratio); the five
quantiles use the normal quantiles scaled by that hour's residual standard
deviation across the pool (an honest spread: too narrow pays for every hour
outside it, too wide pays at every hour).

Why this beats the persistence baseline: a rolling-midnight launch does not stay
flat. Hour 1 (00:00-01:00 UTC) holds only the regions already unlocked
(New Zealand, Australia, Europe, the UK); the United States unlocks at 05:00Z
(East) and 08:00Z (West), Japan at 15:00Z, so viewership climbs through the
day. How far it climbs depends on how big hour 1 already is: the biggest
launches (Elden Ring, Cyberpunk 2077) peak in hour 1 itself, while smaller
hour-1 launches (Resident Evil Requiem, Clair Obscur) rise 5-12x by evening.
That scale dependence is what the ln(v_1) term captures; channels_1 and the
pre-release level carry what scale alone misses.

Evaluated leave-one-out over the 37 past launches in the kit (each launch held
out, fitted on the rest): on the 19 launches most like GTA VI the mean pinball
loss is ~0.25, against ~0.40 for the persistence baseline. Everything the
model uses comes from the case file itself (its history up to the checkpoint
and its comparables); it never reads the target's hours 2-24, never touches
the network, and is fully deterministic.

Python 3.13, standard library only. Run the same way as the kit's starter:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed here, but the line is
# harmless and keeps the starter's contract).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# A comparable joins the shape pool only with a real launch behind it: a
# complete release day, and an hour-1 print of at least this many viewers.
# (Simultaneous-global launches and slow-burn launches have a near-empty hour 1
# and a wildly different shape; they would poison the fit.)
MIN_HOUR1_VIEWERS = 5000
MIN_RELEASE_HOURS = 20
# Tiers the kit's own unlock notes mark as unusable for the shape. Any other
# tier (including one the kit does not name) is kept: the final case adds the
# four practice launches, whose tier labels this program cannot foresee.
EXCLUDED_TIERS = {"early-access-contaminated", "weak", "target-history"}
# The spread never goes below this (log units): the future is never certain.
SPREAD_FLOOR = 0.10
SPREAD_CAP = 2.0
# Ridge penalty for the per-hour fits (on standardised features, slopes only):
# with about eighteen launches per hour and four coefficients, plain least
# squares chases noise; leave-one-out over the kit's launches is best near 3.
RIDGE_PENALTY = 3.0


def _parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def _tier_of(comp):
    unlock = comp.get("unlock", "") or ""
    if "Tier: " in unlock:
        return unlock.split("Tier: ", 1)[1].split(".")[0].split(",")[0].strip()
    return ""


def _release_day(comp):
    """The comparable's release-day hourly viewers, {hour: viewers}, or None.

    hour 1 is [day_start, day_start + 1h); hour h is day_start + (h-1) hours.
    A launch counts as complete when hour 1 exists and at least
    MIN_RELEASE_HOURS of hours 2..24 have a print.
    """
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            when = _parse_time(row["from"])
            viewers = row.get("viewers")
        except (KeyError, ValueError):
            continue
        if viewers is not None:
            by_hour[when] = viewers
    hour1 = by_hour.get(day_start)
    if hour1 is None or hour1 < MIN_HOUR1_VIEWERS:
        return None
    day = {1: hour1}
    for h in range(2, 25):
        value = by_hour.get(day_start + timedelta(hours=h - 1))
        if value is not None:
            day[h] = value
    if len(day) - 1 < MIN_RELEASE_HOURS:
        return None
    return day


def _shape_pool(comparables):
    """The comparables usable for the shape: complete rolling-style launches."""
    pool = []
    for comp in comparables or []:
        if _tier_of(comp) in EXCLUDED_TIERS:
            continue
        day = _release_day(comp)
        if day is not None:
            pool.append((comp, day))
    return pool


def _target_features(case):
    """(p1, ln p1, ln channels_1, ln pre24) for the case at hand.

    pre24 is the median hourly viewers over the 24 hours before the release
    day, from the case's own history_hourly (points before the checkpoint
    only). Falls back to p1 when that window is empty.
    """
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1, first["viewers"])
        channels1 = first.get("channels") or 1
    else:
        history = case.get("history_hourly") or []
        p1 = max(1, history[-1]["viewers"]) if history else 1
        channels1 = (history[-1].get("channels") or 1) if history else 1
    day_start = _parse_time(case["day_start"])
    pre = []
    for row in case.get("history_hourly") or []:
        try:
            when = _parse_time(row["from"])
        except (KeyError, ValueError):
            continue
        if day_start - timedelta(days=1) <= when < day_start:
            if row.get("viewers") is not None:
                pre.append(row["viewers"])
    pre24 = statistics.median(pre) if pre else p1
    return p1, math.log(p1), math.log(max(1, channels1)), math.log(max(1, pre24))


def _pool_features(comp, day):
    """The same three features, for one comparable of the shape pool."""
    day_start = _parse_time(comp["day_start"])
    v1 = max(1, day[1])
    hour1 = next(
        (row for row in comp.get("hourly") or []
         if row.get("from") == comp["day_start"]),
        None,
    )
    channels1 = (hour1 or {}).get("channels") or 1
    pre = []
    for row in comp.get("hourly") or []:
        try:
            when = _parse_time(row["from"])
        except (KeyError, ValueError):
            continue
        if day_start - timedelta(days=1) <= when < day_start:
            if row.get("viewers") is not None:
                pre.append(row["viewers"])
    pre24 = statistics.median(pre) if pre else v1
    return math.log(v1), math.log(max(1, channels1)), math.log(max(1, pre24))


def _solve(design, target, penalty=0.0):
    """Least squares (or ridge) by Gaussian elimination with partial pivoting.

    design: list of rows (each [1, x1, x2, x3]); target: list of y.
    penalty: ridge penalty added to the diagonal for the slope terms only
    (the intercept is never penalised). Returns the coefficient list, or
    None when the system is singular.
    """
    n = len(design[0])
    a = [[sum(row[i] * row[j] for row in design)
          + (penalty if i == j and i > 0 else 0.0)
          for j in range(n)] for i in range(n)]
    b = [sum(row[i] * y for row, y in zip(design, target)) for i in range(n)]
    m = [a[i] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            return None
        for r in range(n):
            if r != col:
                factor = m[r][col] / m[col][col]
                for k in range(col, n + 1):
                    m[r][k] -= factor * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _fit_shape(pool):
    """Per hour, fit ln(v_h / v_1) on the three features over the pool.

    Ridge regression (penalty 3.0) on standardised features: with only about
    eighteen launches per hour and four coefficients, plain least squares
    chases noise; the penalty keeps the fit honest. Returns
    {hour: (coefficients, residual_std, feature_means, feature_stds)} and
    the observed range of ln(v_1) in the pool (for clamping the target's
    scale at predict time).
    """
    rows = {h: [] for h in range(2, 25)}
    for comp, day in pool:
        features = _pool_features(comp, day)
        for h in range(2, 25):
            if h in day:
                rows[h].append((features, math.log(max(1, day[h]) / max(1, day[1]))))
    fitted = {}
    for h in range(2, 25):
        data = rows[h]
        if len(data) < 6:
            fitted[h] = (None, 0.30, None, None)
            continue
        means = [statistics.mean([fv[j] for fv, _ in data]) for j in range(3)]
        stds = [statistics.pstdev([fv[j] for fv, _ in data]) or 1.0 for j in range(3)]
        design = [[1.0] + [(fv[j] - means[j]) / stds[j] for j in range(3)]
                  for fv, _ in data]
        beta = _solve(design, [y for _, y in data], penalty=RIDGE_PENALTY)
        if beta is None:
            fitted[h] = (None, 0.30, None, None)
            continue
        resid = [y - sum(b * x for b, x in zip(beta, row))
                 for row, (_, y) in zip(design, data)]
        fitted[h] = (beta, statistics.pstdev(resid) if len(resid) > 1 else 0.30,
                     means, stds)
    ln_v1 = [math.log(max(1, day[1])) for _, day in pool]
    return fitted, (min(ln_v1), max(ln_v1)) if ln_v1 else (0.0, 0.0)


def _baseline_quantiles(p1, h):
    """The persistence baseline's row (the kit starter's numbers)."""
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def forecast_case(case):
    """The forecast document for one case dict."""
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    p1, lp1, lch1, lpre = _target_features(case)
    pool = _shape_pool(case.get("comparables"))
    hours = []
    if len(pool) < 6:
        # Too little to fit on: fall back to the persistence baseline.
        for row in case["forecast"]:
            h = row["hour"]
            hours.append({"hour": h, "q": [round2(v) for v in _baseline_quantiles(p1, h)]})
        return {"hours": hours}
    fitted, (lo, hi) = _fit_shape(pool)
    lp1 = min(max(lp1, lo), hi)  # never extrapolate the scale past the pool
    for row in case["forecast"]:
        h = row["hour"]
        beta, spread, means, stds = fitted[h]
        if beta is None:
            median = 0.0
        else:
            features = [lp1, lch1, lpre]
            standardised = [(features[j] - means[j]) / stds[j] for j in range(3)]
            median = beta[0] + sum(beta[j + 1] * standardised[j] for j in range(3))
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        quantiles = []
        previous = 0.0
        for z in Z:
            value = p1 * math.exp(median + z * spread)
            value = max(1.0, min(value, 1e9))
            value = max(value, previous)  # non-decreasing
            quantiles.append(round2(value))
            previous = value
        hours.append({"hour": h, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
