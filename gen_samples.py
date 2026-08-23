# -*- coding: utf-8 -*-
"""为 nh3-platform 仓库生成示例曲线 CSV"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.models.renewable import gen_day_data  # noqa: E402

CASES = [
    ("sample_summer_sunny.csv", 7, 6, "sunny", "northwest"),
    ("sample_winter_rain.csv", 11, 12, "rain", "northeast"),
]

for fname, seed, month, weather, region in CASES:
    d = gen_day_data(seed=seed, days=1, season_month=month, region=region, weather=weather)
    lines = ["pv,wind,load"]
    for i in range(96):
        lines.append(f"{float(d['pv'][i]):.1f},{float(d['wind'][i]):.1f},{float(d['load'][i]):.1f}")
    out = os.path.join(ROOT, "samples", fname)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written:", out)
