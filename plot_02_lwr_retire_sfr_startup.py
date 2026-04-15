#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_02_lwr_retire_sfr_startup.py

Reconstruct Figure 4.2 from Bae's notebook as faithfully as possible.

Important
---------
This script intentionally follows the original notebook logic and its quirks,
rather than trying to "improve" the method.

Notebook cells mainly reproduced:
- helper functions from early notebook cells
- deployment extraction cell
- Figure 4.2 plotting cell

Working directory assumed:
    /mnt/c/Users/toh8/Desktop/NEEP_CYCLUS/NC02_MyCyclus/MyCyclus06_B_FENG
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
# NOTEBOOK-LIKE HELPER FUNCTIONS
# =============================================================================
def twosum(timeseries_list):
    """
    Aggregate monthly values into yearly values exactly as in the notebook.

    Despite the name, this does NOT sum pairs.
    It sums 12 monthly values after first removing the first entry.

    The original notebook logic is preserved here intentionally.
    """
    result = [0]
    list_indx = 0
    total = 0
    timeseries_list = list(timeseries_list)[1:]

    while True:
        total += timeseries_list[list_indx]
        list_indx += 1

        if list_indx == len(timeseries_list) - 1:
            break

        if list_indx % 12 == 0:
            result.append(total)
            total = 0

    return result


def read_from_data(data_path):
    """
    Read benchmark data exactly as in the notebook.

    Notebook behavior:
    - start with [0, 0]
    - append one float per line
    - truncate last 4 entries
    """
    with open(data_path, "r") as file:
        timeseries = [0, 0]
        for row in file:
            value = float(row.replace("\n", ""))
            timeseries.append(value)

    return timeseries[:-4]


def push_back_one(timeseries):
    """
    Notebook helper:
    append the last value once, then drop the first value.
    """
    out = np.array(timeseries)
    out = np.append(out, timeseries[-1])
    return out[1:]


# =============================================================================
# SQLITE HELPERS
# =============================================================================
def get_duration(cur):
    """
    Read simulation duration from Info table.
    """
    row = cur.execute("SELECT Duration FROM Info").fetchone()
    if row is None:
        raise RuntimeError("Could not read Duration from Info table.")
    return int(row["Duration"])


def get_monthly_lwr_retirements(cur, duration):
    """
    Reproduce notebook logic for LWR retirements.

    SQL from notebook:
        SELECT exittime FROM agentexit
        INNER JOIN agententry ON agentexit.agentid = agententry.agentid
        WHERE prototype = "lwr"
    """
    lwr_dec = cur.execute(
        """
        SELECT exittime
        FROM agentexit
        INNER JOIN agententry
            ON agentexit.agentid = agententry.agentid
        WHERE prototype = "lwr"
        """
    ).fetchall()

    dec_array = np.zeros(duration)
    exittime_list = []

    for row in lwr_dec:
        exittime_list.append(row["exittime"])

    for i in range(1, duration):
        dec_array[i - 1] = exittime_list.count(i)

    return dec_array


def get_monthly_sfr_startups(cur, duration):
    """
    Reproduce notebook logic for SFR startups.

    SQL from notebook:
        SELECT entertime FROM agententry WHERE prototype = "sfr"
    """
    sfr_dep = cur.execute(
        """
        SELECT entertime
        FROM agententry
        WHERE prototype = "sfr"
        """
    ).fetchall()

    dep_array = np.zeros(duration)
    entertime_list = []

    for row in sfr_dep:
        entertime_list.append(row["entertime"])

    for i in range(1, duration):
        dep_array[i - 1] = entertime_list.count(i)

    return dep_array


# =============================================================================
# MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. Basic checks
    # -------------------------------------------------------------------------
    if not os.path.exists(SQLITE_FILE):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_FILE}")

    if not os.path.exists("./bo/lwr_decom"):
        raise FileNotFoundError("Benchmark file not found: ./bo/lwr_decom")

    if not os.path.exists("./bo/sfr_deploy"):
        raise FileNotFoundError("Benchmark file not found: ./bo/sfr_deploy")

    # -------------------------------------------------------------------------
    # 2. Connect to sqlite
    # -------------------------------------------------------------------------
    conn = sqlite3.connect(SQLITE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # -------------------------------------------------------------------------
    # 3. Match notebook timestep handling
    # -------------------------------------------------------------------------
    duration = get_duration(cur)

    # In the notebook, timestep is effectively an integer sequence of monthly
    # timesteps, not a year-offset array. This is why Figure 4.1 has the odd
    # "2200-3200" style x-axis when plotted directly.
    timestep = np.arange(duration)

    half_length = int((len(timestep) + 1) / 12)
    new_timestep = timestep[:half_length]

    # -------------------------------------------------------------------------
    # 4. Read Cyclus monthly deployment/retirement counts
    # -------------------------------------------------------------------------
    lwr_retired_monthly = get_monthly_lwr_retirements(cur, duration)
    sfr_started_monthly = get_monthly_sfr_startups(cur, duration)

    # -------------------------------------------------------------------------
    # 5. Aggregate Cyclus results exactly like notebook
    # -------------------------------------------------------------------------
    lwr_retired_cyclus = np.array(twosum(lwr_retired_monthly))
    sfr_started_cyclus = np.array(twosum(sfr_started_monthly))

    # -------------------------------------------------------------------------
    # 6. Read benchmark/reference data exactly like notebook
    # -------------------------------------------------------------------------
    data_lwr_decom = read_from_data("./bo/lwr_decom")
    data_sfr_dep = read_from_data("./bo/sfr_deploy")
    data_sfr_dep = push_back_one(data_sfr_dep)

    # -------------------------------------------------------------------------
    # 7. Trim all arrays to common plotting length
    # -------------------------------------------------------------------------
    n = min(
        len(new_timestep),
        len(lwr_retired_cyclus),
        len(sfr_started_cyclus),
        len(data_lwr_decom),
        len(data_sfr_dep),
    )

    x = INIT_YEAR + new_timestep[:n]

    lwr_retired_cyclus = lwr_retired_cyclus[:n]
    sfr_started_cyclus = sfr_started_cyclus[:n]
    data_lwr_decom = np.array(data_lwr_decom[:n])
    data_sfr_dep = np.array(data_sfr_dep[:n])

    # -------------------------------------------------------------------------
    # 8. Plot in notebook order, but control legend explicitly
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # Cyclus bars first, exactly like notebook plotting style
    bar_lwr = plt.bar(
        x,
        lwr_retired_cyclus,
        label="LWRs retired [Cyclus]",
        width=0.8,
    )

    line_lwr, = plt.plot(
        x,
        data_lwr_decom,
        label="LWRs retired [Feng et al.]",
    )

    bar_sfr = plt.bar(
        x,
        sfr_started_cyclus,
        label="SFRs started up [Cyclus]",
        alpha=0.4,
        width=0.8,
    )

    line_sfr, = plt.plot(
        x,
        data_sfr_dep,
        label="SFRs started up [Feng et al.]",
    )

    plt.ylabel("Reactors")
    plt.xlabel("Year")
    plt.title("LWRs retired and SFRs started up each year")
    axes = plt.gca()
    axes.autoscale(tight=True)
    plt.grid(True)
    plt.yticks(np.arange(0, 15, 2))

    # Match thesis-style legend ordering
    handles = [line_lwr, line_sfr, bar_lwr, bar_sfr]
    labels = [
        "LWRs retired [Feng et al.]",
        "SFRs started up [Feng et al.]",
        "LWRs retired [Cyclus]",
        "SFRs started up [Cyclus]",
    ]
    plt.legend(handles, labels)

    if SHOW_FIG:
        plt.show()

    conn.close()


if __name__ == "__main__":
    main()