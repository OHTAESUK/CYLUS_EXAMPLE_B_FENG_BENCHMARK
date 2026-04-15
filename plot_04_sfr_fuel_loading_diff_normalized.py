#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_04_sfr_fuel_loading_diff_normalized.py

Figure 4.4 reconstruction from Bae notebook.

What this figure means
----------------------
This is a diagnostic plot.

It tests whether the difference in annual SFR fresh fuel loading
between CYCLUS and the benchmark can be explained mainly by the fact that:

    CYCLUS SFR core mass  = 15.80 tHM
    Benchmark SFR core mass = 15.63 tHM

Therefore, each newly deployed SFR in CYCLUS requires:

    15.80 - 15.63 = 0.17 tHM

more fresh fuel than the benchmark SFR.

The red line shows:

    (CYCLUS SFR fresh fuel loading - Benchmark SFR fresh fuel loading) / 0.17

So the red line is interpreted as:

    "Equivalent number of extra SFR cores implied by the fuel-loading difference"

The blue bars show the annual number of SFR deployments.

If the red line tracks the blue bars well, then the mismatch is explained mainly by
the SFR core-mass difference caused by integer batch representation in CYCLUS,
not by a mass-flow accounting error.
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

# From Bae thesis Table 4.1
# Benchmark SFR core size = 15.63 tHM
# CYCLUS SFR core size    = 15.80 tHM
SFR_CORE_MASS_DIFF_THM = 0.17


# =============================================================================
# NOTEBOOK-LIKE HELPERS
# =============================================================================
def twosum(timeseries_list):
    """
    Aggregate monthly values into yearly values exactly as in the notebook.

    Important notebook behavior:
    - first remove the first entry
    - then sum every 12 months
    - prepend one zero at the beginning

    This function name comes directly from the notebook, even though it is
    actually doing 12-month aggregation rather than summing two values.
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
    - initialize with [0, 0]
    - append one float per line
    - drop the last 4 entries

    These quirks are preserved intentionally to stay close to the original
    post-processing workflow used in the notebook.
    """
    with open(data_path, "r") as file:
        timeseries = [0, 0]
        for row in file:
            value = float(row.replace("\n", ""))
            timeseries.append(value)

    return timeseries[:-4]


def pull_in_one(timeseries):
    """
    Notebook helper used for aligning annual deployment bars.

    Behavior:
    - prepend the first value once
    - remove the last value

    This effectively shifts the series by one position in the notebook style.
    """
    out = np.array(timeseries)
    out = np.append(timeseries[0], out)
    return out[:-1]


# =============================================================================
# SQLITE / CYCLUS HELPERS
# =============================================================================
def get_duration(cur):
    """
    Read simulation duration from the Info table.
    """
    row = cur.execute("SELECT Duration FROM Info").fetchone()
    if row is None:
        raise RuntimeError("Could not read Duration from Info table.")
    return int(row["Duration"])


def get_timestep_arrays(duration):
    """
    Reconstruct notebook timestep handling.

    Notebook logic:
        timestep = np.arange(duration)
        half_length = int((len(timestep)+1)/12)
        new_timestep = timestep[:half_length]

    The resulting new_timestep is later added to INIT_YEAR.
    """
    timestep = np.arange(duration)
    half_length = int((len(timestep) + 1) / 12)
    new_timestep = timestep[:half_length]
    return timestep, new_timestep


def get_agentids_for_prototype(cur, prototype_name):
    """
    Return all agent IDs belonging to a given prototype.
    """
    rows = cur.execute(
        """
        SELECT agentid
        FROM agententry
        WHERE prototype = ?
        """,
        (prototype_name,),
    ).fetchall()
    return [int(row["agentid"]) for row in rows]


def build_monthly_flux_timeseries(cur, duration, agentids, commodities, inbound=True):
    """
    Build monthly material flux time series for selected agents and commodities.

    Parameters
    ----------
    cur : sqlite cursor
    duration : int
        Simulation duration in months
    agentids : list[int]
        Agent IDs belonging to one prototype
    commodities : list[str]
        Commodity names to track
    inbound : bool
        True  -> track flow INTO these agents (ReceiverId)
        False -> track flow OUT OF these agents (SenderId)

    Returns
    -------
    flux_dict : dict[str, np.ndarray]
        Monthly mass time series by commodity

    Unit conversion
    ---------------
    The sqlite resource quantity here is in kg.
    We convert it to tHM by dividing by 1000.
    """
    flux_dict = {commod: np.zeros(duration, dtype=float) for commod in commodities}

    if len(agentids) == 0:
        return flux_dict

    agent_placeholders = ",".join(["?"] * len(agentids))
    commod_placeholders = ",".join(["?"] * len(commodities))
    agent_col = "ReceiverId" if inbound else "SenderId"

    query = f"""
        SELECT transactions.time AS time,
               transactions.commodity AS commodity,
               resources.quantity AS quantity
        FROM transactions
        INNER JOIN resources
            ON transactions.resourceid = resources.resourceid
        WHERE transactions.{agent_col} IN ({agent_placeholders})
          AND transactions.commodity IN ({commod_placeholders})
    """

    params = list(agentids) + list(commodities)
    rows = cur.execute(query, params).fetchall()

    for row in rows:
        t = int(row["time"])
        c = row["commodity"]
        q = float(row["quantity"]) / 1000.0  # kg -> tHM

        if 0 <= t < duration and c in flux_dict:
            flux_dict[c][t] += q

    return flux_dict


def get_monthly_sfr_startups(cur, duration):
    """
    Reproduce notebook logic for monthly SFR startups.

    SQL source:
        SELECT entertime FROM agententry WHERE prototype = "sfr"
    """
    rows = cur.execute(
        """
        SELECT entertime
        FROM agententry
        WHERE prototype = "sfr"
        """
    ).fetchall()

    dep_array = np.zeros(duration)
    entertime_list = [row["entertime"] for row in rows]

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

    if not os.path.exists("./bo/sfr_fuel_loaded"):
        raise FileNotFoundError("Benchmark file not found: ./bo/sfr_fuel_loaded")

    # -------------------------------------------------------------------------
    # 2. Open sqlite database
    # -------------------------------------------------------------------------
    conn = sqlite3.connect(SQLITE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    duration = get_duration(cur)
    timestep, new_timestep = get_timestep_arrays(duration)

    # -------------------------------------------------------------------------
    # 3. CYCLUS annual SFR fresh fuel loading
    # -------------------------------------------------------------------------
    sfr_agentids = get_agentids_for_prototype(cur, "sfr")
    sfr_load = build_monthly_flux_timeseries(
        cur,
        duration,
        sfr_agentids,
        ["sfr_fuel"],
        inbound=True
    )
    new_sfr_load = np.array(twosum(sfr_load["sfr_fuel"]), dtype=float)

    # -------------------------------------------------------------------------
    # 4. Benchmark annual SFR fresh fuel loading
    # -------------------------------------------------------------------------
    data_sfr_load = np.array(read_from_data("./bo/sfr_fuel_loaded"), dtype=float)

    # -------------------------------------------------------------------------
    # 5. Annual SFR deployment count
    #    Notebook uses pull_in_one(twosum(...)) for alignment.
    # -------------------------------------------------------------------------
    sfr_started_monthly = get_monthly_sfr_startups(cur, duration)
    sfr_started_annual = np.array(twosum(sfr_started_monthly), dtype=float)
    sfr_started_annual = pull_in_one(sfr_started_annual)

    # -------------------------------------------------------------------------
    # 6. Trim all arrays to common plotting length
    # -------------------------------------------------------------------------
    n = min(
        len(new_timestep),
        len(new_sfr_load),
        len(data_sfr_load),
        len(sfr_started_annual),
    )

    x = INIT_YEAR + new_timestep[:n]
    new_sfr_load = new_sfr_load[:n]
    data_sfr_load = data_sfr_load[:n]
    sfr_started_annual = sfr_started_annual[:n]

    # -------------------------------------------------------------------------
    # 7. Compute normalized difference
    # -------------------------------------------------------------------------
    diff_sfr_load = new_sfr_load - data_sfr_load
    norm_diff_sfr_load = diff_sfr_load / SFR_CORE_MASS_DIFF_THM

    # -------------------------------------------------------------------------
    # 8. Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Figure 4.4 diagnostic summary")
    print("=" * 78)
    print(f"SQLite file used                 : {SQLITE_FILE}")
    print(f"SFR core mass difference used   : {SFR_CORE_MASS_DIFF_THM:.2f} tHM")
    print()
    print("Interpretation:")
    print("  Red line = (CYCLUS SFR fresh fuel - Benchmark SFR fresh fuel) / 0.17")
    print("  Blue bars = annual number of SFR deployments")
    print()
    print("Core message:")
    print("  If the red line follows the blue bars, then the annual SFR fuel-loading")
    print("  mismatch is explained mainly by the fact that each CYCLUS SFR core is")
    print("  0.17 tHM larger than the benchmark SFR core.")
    print()
    print("  In other words, this figure is a diagnostic check showing that the")
    print("  mismatch comes primarily from integer-batch core representation in")
    print("  CYCLUS, rather than from a material-flow accounting error.")
    print("=" * 78 + "\n")

    # -------------------------------------------------------------------------
    # 9. Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))

    # Red line: normalized equivalent number of extra SFR cores
    plt.plot(
        x,
        norm_diff_sfr_load,
        label="Normalized SFR fuel load difference\n[(CYCLUS - Benchmark) / 0.17 tHM per SFR core]",
        color="red",
        linewidth=2.0,
    )

    # Blue bars: annual SFR deployment count
    plt.bar(
        x,
        sfr_started_annual,
        label="Annual SFR deployment count [reactors/year]",
        width=0.8,
        alpha=0.85,
    )

    plt.xlabel("Year")
    plt.ylabel("Equivalent number of SFR cores / Deployment count [reactors]")
    plt.title(
        "Figure 4.4-style diagnostic:\n"
        "Annual SFR fuel-loading mismatch normalized by the SFR core-mass difference"
    )
    plt.grid(True)
    plt.legend()

    # Add a compact explanatory note inside the figure
    plt.text(
        0.02,
        0.02,
        "Red line near blue bars => mismatch mainly explained by\n"
        "CYCLUS SFR core being 0.17 tHM larger than benchmark SFR core.",
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if SHOW_FIG:
        plt.show()

    conn.close()


if __name__ == "__main__":
    main()