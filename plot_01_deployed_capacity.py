#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4.1 reconstruction
Deployed reactor capacities at the end of each year
Spyder-friendly, display only (no automatic file saving)
"""

import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# USER INPUT
# =============================================================================
SQLITE_FILE = "output_Original.sqlite"
INIT_YEAR = 2015
SHOW_FIG = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_duration(cur):
    row = cur.execute("SELECT Duration FROM Info").fetchone()
    return int(row[0])


def get_power_series(cur, prototype_name, duration):
    """
    Reconstruct deployed capacity by prototype.
    Capacity is treated as existing from EnterTime to EnterTime + Lifetime.
    Monthly deployed capacity is built directly from AgentEntry.
    """
    rows = cur.execute(
        """
        SELECT EnterTime, Lifetime
        FROM AgentEntry
        WHERE Prototype = ?
        """,
        (prototype_name,),
    ).fetchall()

    power_map = {
        "lwr": 1.0,       # 1000 MWe -> 1.0 GWe
        "sfr": 0.3333,    # 333.3 MWe -> 0.3333 GWe
    }

    if prototype_name not in power_map:
        raise ValueError(f"Unknown prototype_name: {prototype_name}")

    unit_power = power_map[prototype_name]
    ts = np.zeros(duration, dtype=float)

    for enter_time, lifetime in rows:
        enter_time = int(enter_time)
        lifetime = int(lifetime)
        exit_time = min(duration, enter_time + lifetime)
        ts[enter_time:exit_time] += unit_power

    return ts


def annual_end_snapshot(monthly_ts):
    """
    End-of-year snapshot:
    take the value at month index 11, 23, 35, ...
    """
    annual = []
    for i in range(11, len(monthly_ts), 12):
        annual.append(monthly_ts[i])
    return np.array(annual, dtype=float)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(SQLITE_FILE):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_FILE}")

    conn = sqlite3.connect(SQLITE_FILE)
    cur = conn.cursor()

    duration = get_duration(cur)

    lwr_monthly = get_power_series(cur, "lwr", duration)
    sfr_monthly = get_power_series(cur, "sfr", duration)

    lwr_annual = annual_end_snapshot(lwr_monthly)
    sfr_annual = annual_end_snapshot(sfr_monthly)

    n_years = min(len(lwr_annual), len(sfr_annual))
    lwr_annual = lwr_annual[:n_years]
    sfr_annual = sfr_annual[:n_years]
    years = INIT_YEAR + np.arange(n_years)

    plt.figure(figsize=(7, 5))

    plt.bar(
        years,
        lwr_annual,
        width=1.0,
        label="lwr_inst",
    )

    plt.bar(
        years,
        sfr_annual,
        width=1.0,
        bottom=lwr_annual,
        label="sfr_inst",
    )

    plt.xlabel("Year")
    plt.ylabel("Deployed Capacity [GWe]")
    plt.title("Deployed reactor capacities at the end of each year.")
    plt.grid(True, alpha=0.4)
    plt.legend()

    if SHOW_FIG:
        plt.show()

    conn.close()


if __name__ == "__main__":
    main()