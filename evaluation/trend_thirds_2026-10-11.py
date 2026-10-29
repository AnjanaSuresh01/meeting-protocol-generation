# -*- coding: utf-8 -*-
"""Runs an exploratory Cochran-Armitage trend test over the thirds of a meeting (first, middle, last).

Reads the "recorded" and "n" counts of each third from goldset/thirds_recall_2026-10-10.json (by_third),
a path relative to the working directory, scores the thirds 0, 1 and 2, and prints the counts, z and
the two-sided p-value.

    python trend_thirds_2026-10-11.py    # run from the folder that holds goldset/

Prints to stdout only.
"""
import json
from math import erf, sqrt

d = json.load(open("goldset/thirds_recall_2026-10-10.json", encoding="utf-8"))["by_third"]
x = [d[k]["recorded"] for k in ("first", "middle", "last")]
n = [d[k]["n"] for k in ("first", "middle", "last")]
w = [0, 1, 2]
N, X = sum(n), sum(x)
p = X / N
T = sum(wi * (xi - ni * p) for wi, xi, ni in zip(w, x, n))
V = p * (1 - p) * (sum(ni * wi * wi for ni, wi in zip(n, w)) - sum(ni * wi for ni, wi in zip(n, w)) ** 2 / N)
z = T / sqrt(V)
print(f"counts {x} of {n}; z = {z:.2f}; two-sided p = {2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))):.3f}")
