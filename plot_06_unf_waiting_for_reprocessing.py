#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_06_unf_waiting_for_reprocessing.py

Purpose
=======
Reconstruct Figure 4.6:
    "Inventory of discharge and cooled UNF waiting for reprocessing"

using the user's CURRENT ordinary Cyclus sqlite output:
    output_Original.sqlite

without requiring the special "standardized_verif" Cycamore branch.

Why this script exists
======================
In the original notebook, Figure 4.6 relies on extra sqlite tables such as:

    - storageinventory
    - separationevents

Those tables are available in the special standardized_verif setup,
but they are NOT guaranteed to exist in the user's current sqlite.

Therefore, this script does the following:

1) If the special tables exist, use them directly.
2) Otherwise, reconstruct equivalent quantities from ordinary
   Transactions + Resources + AgentEntry tables.

Final logic that matched reasonably well
========================================
The version that visually matched the benchmark "well enough" used:

LWR:
----
- "Now Cooled" monthly amount:
    reconstructed from monthly inflow to uox_unf_storage,
    shifted forward by fixed cooling residence time.

- monthly reprocessed amount:
    if separationevents exists, use it directly and apply pull_in_one(...)
    exactly like the notebook.
    otherwise, approximate LWR reprocessing throughput by:
        TOTAL monthly outflow from uox_reprocessing
        excluding feed-like commodities.

- monthly waiting inventory:
    cumulative sum of
        (monthly now cooled) - (monthly reprocessed throughput)

This is the key notebook meaning for LWR.

SFR:
----
- "Now Cooled" monthly amount:
    reconstructed from monthly inflow to sfr_unf_storage,
    shifted forward by fixed cooling residence time.

- notebook does NOT use separationevents for the final SFR waiting curve.
  Instead, it computes:
      annual now cooled - annual cooled_sfr_unf sent to sfr_reprocessing

So for SFR we preserve that notebook style:
    waiting_annual = annual_now_cooled - annual_feed_to_reprocessing

Important note
==============
This script intentionally keeps LWR and SFR logic DIFFERENT because the
original notebook itself treats them differently in Figure 4.6.

Display behavior
================
This script only shows the figure.
It does NOT save the figure automatically.
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

# -----------------------------------------------------------------------------
# Cooling completion timing reconstruction
# -----------------------------------------------------------------------------
# These values are for reconstructing the monthly amount that becomes
# "Now Cooled" in Figure 4.6 when storageinventory is absent.
#
# Why 47 and 11 instead of 48 and 12?
# -----------------------------------
# In the earlier Figure 4.5 storage-inventory interpretation, one may think of
# the residence as 48 and 12 months respectively.
#
# However, for Figure 4.6 we are reconstructing the MONTHLY AMOUNT that FINISHES
# cooling at each timestep, i.e. the notebook's "Now Cooled" concept.
#
# Empirically, for this figure the correct timing alignment is better matched by:
#   LWR: shift by 47 months
#   SFR: shift by 11 months
#
# This is a timestep-alignment choice for reproducing Figure 4.6 behavior.
LWR_NOW_COOLED_SHIFT = 47
SFR_NOW_COOLED_SHIFT = 11

# -----------------------------------------------------------------------------
# Prototype names expected in the user's Cyclus sqlite
# -----------------------------------------------------------------------------
LWR_STORAGE_PROTO = "uox_unf_storage"
SFR_STORAGE_PROTO = "sfr_unf_storage"

LWR_REPROC_PROTO = "uox_reprocessing"
SFR_REPROC_PROTO = "sfr_reprocessing"

# -----------------------------------------------------------------------------
# Candidate commodity names for inflow to storage
# -----------------------------------------------------------------------------
# We use candidate lists rather than one fixed string because the user's sqlite
# may use a slightly different commodity naming convention.
#
# The script will inspect the ACTUAL inbound commodities to the prototype and
# choose the first candidate that exists.
LWR_STORAGE_IN_COMMOD_CANDIDATES = [
    "uox_unf",
]

SFR_STORAGE_IN_COMMOD_CANDIDATES = [
    "sfr_unf",
]

# -----------------------------------------------------------------------------
# Candidate commodity names for SFR feed into reprocessing
# -----------------------------------------------------------------------------
# In the original notebook Figure 4.6, SFR waiting is computed using
# cooled_sfr_unf sent to sfr_reprocessing.
#
# If the exact name is not present, we allow fallback to "sfr_unf".
SFR_REPROC_IN_COMMOD_CANDIDATES = [
    "cooled_sfr_unf",
    "sfr_unf",
]

# -----------------------------------------------------------------------------
# LWR reprocessing fallback exclusion rules
# -----------------------------------------------------------------------------
# If separationevents does not exist, we reconstruct LWR reprocessing throughput
# from TOTAL monthly OUTFLOW from the LWR reprocessing facility.
#
# However, feed-like commodities should obviously NOT be counted as processed
# output. These are excluded here.
LWR_REPROC_FEED_COMMODS = {
    "cooled_uox_unf",
    "uox_unf",
}

# Optional additional outbound commodities to exclude from the throughput sum.
# Leave empty unless you inspect the sqlite and find some outbound flow that is
# not an actual processed product/waste stream.
LWR_REPROC_EXCLUDE_OUTBOUND = set()


# =============================================================================
# NOTEBOOK-LIKE HELPER FUNCTIONS
# =============================================================================
def twosum(timeseries_list):
    """
    Aggregate monthly values into yearly values exactly as in the notebook.

    Important notebook behavior
    ---------------------------
    Despite the name "twosum", this does NOT sum pairs.
    The original notebook does the following:
        1) remove the first monthly entry
        2) sum each 12-month block
        3) prepend a leading zero

    We preserve that behavior exactly because the benchmark comparison curves
    and the notebook's x-axis alignment depend on these quirks.
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

    Notebook behavior
    -----------------
    The benchmark text files are read with the following quirks:
        - start with [0, 0]
        - append one float per line
        - drop the last 4 entries

    We intentionally preserve this behavior for consistency with the original
    verification notebook.
    """
    with open(data_path, "r") as file:
        timeseries = [0, 0]
        for row in file:
            value = float(row.replace("\n", ""))
            timeseries.append(value)
    return timeseries[:-4]


def pull_in_one(timeseries):
    """
    Notebook helper used in LWR reprocessing alignment.

    Behavior
    --------
    prepend the first value once, then drop the last value.

    Why this matters
    ----------------
    In the original Figure 4.6 notebook cell, the LWR reprocessing monthly
    throughput is shifted using pull_in_one(...). This is one of the notebook
    timing quirks that materially affects the alignment of the LWR waiting curve.

    We keep this exactly for LWR.
    """
    out = np.array(timeseries, dtype=float)
    out = np.append(timeseries[0], out)
    return out[:-1]


def timeseries(specific_search, duration, kg_to_tons):
    """
    Same semantic meaning as analysis.py timeseries(...)

    Parameters
    ----------
    specific_search : list of (time, value)
        Example:
            [(0, 1000.0), (0, 500.0), (1, 700.0), ...]
        The function sums all values belonging to the same month index.

    duration : int
        Total number of months in the simulation.

    kg_to_tons : bool
        If True, convert kg to tHM by multiplying by 0.001.

    Returns
    -------
    np.ndarray
        Monthly time series of length = duration.

    Why this helper is here
    -----------------------
    The original notebook depends heavily on analysis.py helper semantics.
    Re-implementing the same meaning locally keeps the script self-contained.
    """
    value_timeseries = []
    array = np.array(specific_search, dtype=float)

    if len(specific_search) > 0:
        for i in range(0, duration):
            value = np.sum(array[array[:, 0] == i][:, 1])
            if kg_to_tons:
                value_timeseries.append(value * 0.001)
            else:
                value_timeseries.append(value)
    else:
        value_timeseries = [0.0] * duration

    return np.array(value_timeseries, dtype=float)


# =============================================================================
# SQLITE HELPER FUNCTIONS
# =============================================================================
def get_duration(cur):
    """
    Read the total simulation duration (in months) from Info table.
    """
    row = cur.execute("SELECT Duration FROM Info").fetchone()
    if row is None:
        raise RuntimeError("Could not read Duration from Info table.")
    return int(row["Duration"])


def get_timestep_arrays(duration):
    """
    Reconstruct notebook timestep handling.

    Returns
    -------
    timestep : np.ndarray
        Monthly integer timesteps: [0, 1, 2, ..., duration-1]

    new_timestep : np.ndarray
        Notebook-style yearly x-axis helper created by slicing monthly timestep
        before annual aggregation.
    """
    timestep = np.arange(duration)
    half_length = int((len(timestep) + 1) / 12)
    new_timestep = timestep[:half_length]
    return timestep, new_timestep


def table_exists(cur, table_name):
    """
    Check whether a given sqlite table exists.
    """
    row = cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def prototype_id(cur, prototype):
    """
    Return all agent IDs whose prototype matches the requested prototype name.
    """
    rows = cur.execute(
        """
        SELECT agentid
        FROM agententry
        WHERE prototype = ?
        COLLATE NOCASE
        """,
        (prototype,),
    ).fetchall()
    return [int(row["agentid"]) for row in rows]


def distinct_inbound_commodities(cur, prototype_name):
    """
    Return the sorted distinct inbound commodity names for a prototype.

    Why useful
    ----------
    Ordinary sqlite outputs may not always use exactly the commodity string
    we expect. This helper lets the script discover what is actually flowing
    into the facility.
    """
    ids = prototype_id(cur, prototype_name)
    if len(ids) == 0:
        return []

    placeholders = ",".join(["?"] * len(ids))
    rows = cur.execute(
        f"""
        SELECT DISTINCT commodity
        FROM transactions
        WHERE receiverid IN ({placeholders})
        ORDER BY commodity
        """,
        ids,
    ).fetchall()

    return [row["commodity"] for row in rows]


def distinct_outbound_commodities(cur, prototype_name):
    """
    Return the sorted distinct outbound commodity names for a prototype.

    Why useful
    ----------
    For LWR fallback, we reconstruct reprocessing throughput from total
    outbound mass from the reprocessing facility. Therefore we need to know
    which commodity streams actually leave that facility.
    """
    ids = prototype_id(cur, prototype_name)
    if len(ids) == 0:
        return []

    placeholders = ",".join(["?"] * len(ids))
    rows = cur.execute(
        f"""
        SELECT DISTINCT commodity
        FROM transactions
        WHERE senderid IN ({placeholders})
        ORDER BY commodity
        """,
        ids,
    ).fetchall()

    return [row["commodity"] for row in rows]


def first_existing_inbound_commodity(cur, prototype_name, candidates):
    """
    Choose the first candidate commodity name that actually exists as inbound
    flow to the given prototype.

    This is a robust way to handle slight naming differences between sqlite
    outputs while still keeping the intended logic explicit.
    """
    available = set(distinct_inbound_commodities(cur, prototype_name))
    for name in candidates:
        if name in available:
            return name

    raise RuntimeError(
        f'No matching inbound commodity found for "{prototype_name}". '
        f'Available inbound commodities: {sorted(available)}'
    )


def monthly_inflow_to_prototype(cur, duration, prototype_name, commodity_name):
    """
    Sum monthly quantity [tHM] flowing INTO all agents of a prototype
    for one specified commodity.

    Data source
    -----------
    Transactions.receiverid + Resources.quantity

    Unit conversion
    ---------------
    Resources.quantity is assumed to be kg, and is converted to tHM here.
    """
    ids = prototype_id(cur, prototype_name)
    monthly = np.zeros(duration, dtype=float)

    if len(ids) == 0:
        raise RuntimeError(f'Prototype "{prototype_name}" not found in AgentEntry.')

    placeholders = ",".join(["?"] * len(ids))
    rows = cur.execute(
        f"""
        SELECT transactions.time AS time,
               resources.quantity AS quantity
        FROM transactions
        INNER JOIN resources
            ON transactions.resourceid = resources.resourceid
        WHERE transactions.receiverid IN ({placeholders})
          AND transactions.commodity = ?
        """,
        ids + [commodity_name],
    ).fetchall()

    for row in rows:
        t = int(row["time"])
        q = float(row["quantity"]) * 0.001  # kg -> tHM
        if 0 <= t < duration:
            monthly[t] += q

    return monthly


def monthly_outflow_to_prototype(cur, duration, prototype_name, commodity_name):
    """
    Sum monthly quantity [tHM] flowing OUT OF all agents of a prototype
    for one specified commodity.

    Data source
    -----------
    Transactions.senderid + Resources.quantity
    """
    ids = prototype_id(cur, prototype_name)
    monthly = np.zeros(duration, dtype=float)

    if len(ids) == 0:
        raise RuntimeError(f'Prototype "{prototype_name}" not found in AgentEntry.')

    placeholders = ",".join(["?"] * len(ids))
    rows = cur.execute(
        f"""
        SELECT transactions.time AS time,
               resources.quantity AS quantity
        FROM transactions
        INNER JOIN resources
            ON transactions.resourceid = resources.resourceid
        WHERE transactions.senderid IN ({placeholders})
          AND transactions.commodity = ?
        """,
        ids + [commodity_name],
    ).fetchall()

    for row in rows:
        t = int(row["time"])
        q = float(row["quantity"]) * 0.001  # kg -> tHM
        if 0 <= t < duration:
            monthly[t] += q

    return monthly


def monthly_storage_now_cooled_from_table(cur, duration, prototype_name):
    """
    Direct notebook path if storageinventory exists.

    Meaning
    -------
    This returns the monthly amount labeled as status='Now Cooled'
    for the storage facility, converted to tHM.

    This is the closest possible reproduction of the original notebook logic.
    """
    rows = cur.execute(
        """
        SELECT time, quantity
        FROM storageinventory
        WHERE prototype = ?
          AND status = 'Now Cooled'
        """,
        (prototype_name,),
    ).fetchall()

    pairs = [(int(row["time"]), float(row["quantity"])) for row in rows]
    return timeseries(pairs, duration, True)


def monthly_reprocessed_from_separationevents(cur, duration, prototype_name):
    """
    Direct notebook path if separationevents exists.

    Meaning
    -------
    This returns the monthly UNF reprocessing throughput from the special
    separationevents table.

    This is exactly what the notebook uses for LWR throughput.
    """
    ids = prototype_id(cur, prototype_name)
    if len(ids) == 0:
        raise RuntimeError(f'Prototype "{prototype_name}" not found in AgentEntry.')

    rows = cur.execute(
        """
        SELECT time, value
        FROM separationevents
        WHERE type = 'UNF'
          AND agentid = ?
        """,
        (ids[0],),
    ).fetchall()

    pairs = [(int(row["time"]), float(row["value"])) for row in rows]
    return timeseries(pairs, duration, True)


def shift_by_months(monthly_inflow, shift_months):
    """
    Reconstruct the monthly amount that FINISHES cooling at each month.

    Interpretation
    --------------
    If fuel enters storage at month t and must remain there N months,
    then it becomes "Now Cooled" at t + N.

    So this helper shifts storage inflow forward by a fixed number of months.
    """
    out = np.zeros_like(monthly_inflow, dtype=float)
    if shift_months < len(monthly_inflow):
        out[shift_months:] = monthly_inflow[:-shift_months]
    return out


def reconstruct_now_cooled_from_storage_inflow(
    cur,
    duration,
    storage_proto,
    storage_in_commodity,
    cooling_months,
):
    """
    Reconstruct monthly "Now Cooled" amount from ordinary storage inflow.

    This is the fallback path when storageinventory does not exist.
    """
    inflow = monthly_inflow_to_prototype(
        cur=cur,
        duration=duration,
        prototype_name=storage_proto,
        commodity_name=storage_in_commodity,
    )
    return shift_by_months(inflow, cooling_months)


def reconstruct_lwr_reprocessing_throughput_from_outflows(cur, duration, prototype_name):
    """
    Fallback reconstruction for LWR monthly reprocessing throughput.

    Why this fallback is used
    -------------------------
    In the notebook, LWR throughput comes from separationevents(type='UNF').
    If separationevents does not exist in the current sqlite, then using
    reprocessing INFLOW as a proxy turns out to be too crude and leaves a large
    unrealistic sawtooth in the waiting inventory.

    The better approximation is:
        total monthly OUTFLOW from uox_reprocessing,
        excluding feed-like commodities.

    Why this is more reasonable
    ---------------------------
    The monthly amount that a reprocessing facility PROCESSES is better
    represented by what the facility actually sends out as products/wastes
    than by what simply arrives at its input port.

    This fallback was the key change that made the LWR curve look plausible.
    """
    out_comms = distinct_outbound_commodities(cur, prototype_name)

    use_comms = [
        c for c in out_comms
        if c not in LWR_REPROC_FEED_COMMODS
        and c not in LWR_REPROC_EXCLUDE_OUTBOUND
    ]

    if len(use_comms) == 0:
        raise RuntimeError(
            f'No outbound product commodities found for "{prototype_name}". '
            f'Available outbound commodities: {out_comms}'
        )

    total = np.zeros(duration, dtype=float)
    for comm in use_comms:
        total += monthly_outflow_to_prototype(
            cur=cur,
            duration=duration,
            prototype_name=prototype_name,
            commodity_name=comm,
        )

    print(f"[INFO] LWR throughput fallback outbound commodities = {use_comms}")
    return total


# =============================================================================
# MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. Basic file checks
    # -------------------------------------------------------------------------
    if not os.path.exists(SQLITE_FILE):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_FILE}")

    for path in ("./bo/lwr_cooled", "./bo/sfr_cooled"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Benchmark file not found: {path}")

    # -------------------------------------------------------------------------
    # 2. Open sqlite and collect basic run metadata
    # -------------------------------------------------------------------------
    conn = sqlite3.connect(SQLITE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    duration = get_duration(cur)
    timestep, new_timestep = get_timestep_arrays(duration)

    has_storageinventory = table_exists(cur, "storageinventory")
    has_separationevents = table_exists(cur, "separationevents")

    print(f"[INFO] duration = {duration}")
    print(f"[INFO] storageinventory exists? {has_storageinventory}")
    print(f"[INFO] separationevents exists? {has_separationevents}")

    # -------------------------------------------------------------------------
    # 3. LWR storage commodity selection
    # -------------------------------------------------------------------------
    # We determine which commodity actually flows into uox_unf_storage.
    lwr_storage_commod = first_existing_inbound_commodity(
        cur,
        LWR_STORAGE_PROTO,
        LWR_STORAGE_IN_COMMOD_CANDIDATES,
    )
    print(f"[INFO] LWR storage inbound commodity = {lwr_storage_commod}")

    # -------------------------------------------------------------------------
    # 4. LWR monthly "Now Cooled"
    # -------------------------------------------------------------------------
    # Direct path if special table exists; otherwise reconstruct by shifting
    # storage inflow by fixed cooling residence time.
    if has_storageinventory:
        lwr_now_cooled_month = monthly_storage_now_cooled_from_table(
            cur, duration, LWR_STORAGE_PROTO
        )
        print("[INFO] LWR now-cooled amount: using storageinventory")
    else:
        lwr_now_cooled_month = reconstruct_now_cooled_from_storage_inflow(
            cur=cur,
            duration=duration,
            storage_proto=LWR_STORAGE_PROTO,
            storage_in_commodity=lwr_storage_commod,
            cooling_months=LWR_NOW_COOLED_SHIFT,
        )
        print(
            f"[INFO] LWR now-cooled amount: reconstructed from storage inflow shift "
            f"(shift={LWR_NOW_COOLED_SHIFT})"
        )

    # -------------------------------------------------------------------------
    # 5. LWR monthly reprocessing throughput
    # -------------------------------------------------------------------------
    # If separationevents exists, use notebook logic directly.
    # Otherwise use total outbound mass from uox_reprocessing as the fallback.
    #
    # IMPORTANT:
    # After either path, apply pull_in_one(...), because notebook applies it
    # to LWR reprocessing monthly throughput.
    if has_separationevents:
        lwr_rep_month = monthly_reprocessed_from_separationevents(
            cur, duration, LWR_REPROC_PROTO
        )
        lwr_rep_month = pull_in_one(lwr_rep_month)
        print("[INFO] LWR reprocessed amount: using separationevents + pull_in_one")
    else:
        lwr_rep_month = reconstruct_lwr_reprocessing_throughput_from_outflows(
            cur=cur,
            duration=duration,
            prototype_name=LWR_REPROC_PROTO,
        )
        lwr_rep_month = pull_in_one(lwr_rep_month)
        print("[INFO] LWR reprocessed amount: using TOTAL outbound mass + pull_in_one")

    # -------------------------------------------------------------------------
    # 6. LWR waiting-for-reprocessing inventory (monthly)
    # -------------------------------------------------------------------------
    # This is the notebook meaning:
    #
    #   cooled_inv[t] = cooled_inv[t-1] + now_cooled[t] - reprocessed[t]
    #
    # This is a cumulative mass-balance style inventory of fuel that HAS
    # completed mandatory cooling but HAS NOT yet been reprocessed.
    lwr_waiting_month = np.zeros(duration, dtype=float)
    for t in range(1, duration):
        lwr_waiting_month[t] = (
            lwr_waiting_month[t - 1]
            + lwr_now_cooled_month[t]
            - lwr_rep_month[t]
        )

    # -------------------------------------------------------------------------
    # 7. SFR storage commodity selection
    # -------------------------------------------------------------------------
    sfr_storage_commod = first_existing_inbound_commodity(
        cur,
        SFR_STORAGE_PROTO,
        SFR_STORAGE_IN_COMMOD_CANDIDATES,
    )
    print(f"[INFO] SFR storage inbound commodity = {sfr_storage_commod}")

    # -------------------------------------------------------------------------
    # 8. SFR monthly "Now Cooled"
    # -------------------------------------------------------------------------
    # Same reconstruction idea as LWR, but using SFR storage and SFR timing.
    if has_storageinventory:
        sfr_now_cooled_month = monthly_storage_now_cooled_from_table(
            cur, duration, SFR_STORAGE_PROTO
        )
        print("[INFO] SFR now-cooled amount: using storageinventory")
    else:
        sfr_now_cooled_month = reconstruct_now_cooled_from_storage_inflow(
            cur=cur,
            duration=duration,
            storage_proto=SFR_STORAGE_PROTO,
            storage_in_commodity=sfr_storage_commod,
            cooling_months=SFR_NOW_COOLED_SHIFT,
        )
        print(
            f"[INFO] SFR now-cooled amount: reconstructed from storage inflow shift "
            f"(shift={SFR_NOW_COOLED_SHIFT})"
        )

    # Notebook compares SFR waiting on an annual basis in this figure.
    sfr_now_cooled_annual = np.array(twosum(sfr_now_cooled_month), dtype=float)

    # -------------------------------------------------------------------------
    # 9. SFR annual waiting
    # -------------------------------------------------------------------------
    # This intentionally follows notebook style:
    #
    #   sfr_waiting = annual(now_cooled) - annual(cooled_sfr_unf sent to reprocessing)
    #
    # In other words, for SFR we do NOT use separationevents for the final
    # waiting curve in Figure 4.6. We use the feed sent into sfr_reprocessing.
    sfr_reproc_feed = first_existing_inbound_commodity(
        cur,
        SFR_REPROC_PROTO,
        SFR_REPROC_IN_COMMOD_CANDIDATES,
    )
    sfr_fuel_sent_month = monthly_inflow_to_prototype(
        cur=cur,
        duration=duration,
        prototype_name=SFR_REPROC_PROTO,
        commodity_name=sfr_reproc_feed,
    )
    print(f"[INFO] SFR reprocessing feed commodity = {sfr_reproc_feed}")

    sfr_fuel_sent_annual = np.array(twosum(sfr_fuel_sent_month), dtype=float)
    sfr_waiting_annual = sfr_now_cooled_annual - sfr_fuel_sent_annual

    # -------------------------------------------------------------------------
    # 10. Read benchmark curves from bo/ files
    # -------------------------------------------------------------------------
    data_lwr_waiting = np.array(read_from_data("./bo/lwr_cooled"), dtype=float)
    data_sfr_waiting = np.array(read_from_data("./bo/sfr_cooled"), dtype=float)

    # -------------------------------------------------------------------------
    # 11. Trim arrays for plotting
    # -------------------------------------------------------------------------
    # LWR Cyclus curve is monthly.
    # benchmark LWR curve is annual notebook-style.
    #
    # SFR Cyclus curve is annual notebook-style.
    # benchmark SFR curve is annual notebook-style.
    n_monthly = min(len(timestep), len(lwr_waiting_month))
    x_lwr_month = INIT_YEAR + timestep[:n_monthly] / 12.0
    lwr_waiting_month = lwr_waiting_month[:n_monthly]

    n_annual = min(
        len(new_timestep),
        len(sfr_waiting_annual),
        len(data_lwr_waiting),
        len(data_sfr_waiting),
    )
    x_annual = INIT_YEAR + new_timestep[:n_annual]

    sfr_waiting_annual = sfr_waiting_annual[:n_annual]
    data_lwr_waiting = data_lwr_waiting[:n_annual]
    data_sfr_waiting = data_sfr_waiting[:n_annual]

    # -------------------------------------------------------------------------
    # 12. Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(x_lwr_month, lwr_waiting_month, label="LWR UNF [Cyclus]")
    plt.plot(x_annual, data_lwr_waiting, label="LWR UNF [Feng et al.]", linestyle="-.")
    plt.plot(x_annual, sfr_waiting_annual, label="SFR UNF [Cyclus]")
    plt.plot(x_annual, data_sfr_waiting, label="SFR UNF [Feng et al.]", linestyle="-.")

    plt.xlabel("Year")
    plt.ylabel("UNF waiting for Reprocessing [MTHM]")
    plt.title("Inventory of discharge and cooled UNF waiting for reprocessing")
    plt.legend()

    axes = plt.gca()
    #axes.autoscale(tight=True)
    plt.grid(True)

    if SHOW_FIG:
        plt.show()

    conn.close()


if __name__ == "__main__":
    main()