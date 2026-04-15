#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_07_annual_reprocessing_throughputs.py

Figure 4.7 reconstruction
Annual reprocessing throughputs

Reference notebook logic
------------------------
Notebook cell logic is:

LWR:
    uox_rep_agentid = an.prototype_id(cur, 'uox_reprocessing')
    uox_rep = cur.execute('SELECT Time, Value from separationevents '
                          'WHERE Type = "UNF" AND Agentid = ' + uox_rep_agentid[0]).fetchall()
    uox_rep = twosum(an.timeseries(uox_rep, duration, True))
    uox_rep = pull_in_one(uox_rep)

SFR:
    sfr_rep_agentid = an.prototype_id(cur, 'sfr_reprocessing')
    sfr_rep = cur.execute('SELECT Time, Value from separationevents '
                          'WHERE Type = "UNF" AND Agentid = ' + sfr_rep_agentid[0]).fetchall()
    sfr_rep = twosum(an.timeseries(sfr_rep, duration, True))

So:
- LWR annual throughput gets pull_in_one(...)
- SFR annual throughput does NOT get pull_in_one(...)

Current sqlite compatibility
----------------------------
If separationevents exists:
    use notebook logic directly

If separationevents does NOT exist:
    use minimal fallback based on the working Figure 4.6 reconstruction:
    - LWR throughput proxy:
        total monthly OUTFLOW from uox_reprocessing excluding feed commodities
    - SFR throughput proxy:
        monthly INFLOW of cooled_sfr_unf (or fallback candidate) into sfr_reprocessing

This script does NOT automatically save the figure.
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

LWR_REPROC_PROTO = "uox_reprocessing"
SFR_REPROC_PROTO = "sfr_reprocessing"

# Candidate SFR feed names for fallback
SFR_REPROC_IN_COMMOD_CANDIDATES = [
    "cooled_sfr_unf",
    "sfr_unf",
]

# LWR feed-like commodities to exclude from total reprocessing outflow fallback
LWR_REPROC_FEED_COMMODS = {
    "cooled_uox_unf",
    "uox_unf",
}

# Optional extra outbound commodities to exclude from LWR fallback
LWR_REPROC_EXCLUDE_OUTBOUND = set()


# =============================================================================
# NOTEBOOK-LIKE HELPERS
# =============================================================================
def twosum(timeseries_list):
    """
    Aggregate monthly values into yearly values exactly as in the notebook.

    Notebook behavior:
    - remove first entry
    - sum each 12-month block
    - prepend leading zero
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
    """
    with open(data_path, "r") as file:
        timeseries = [0, 0]
        for row in file:
            value = float(row.replace("\n", ""))
            timeseries.append(value)
    return timeseries[:-4]


def pull_in_one(timeseries):
    """
    Notebook helper:
    prepend the first value once, then remove the last value.

    In Figure 4.7 notebook logic, this is applied to LWR annual throughput only.
    """
    out = np.array(timeseries, dtype=float)
    out = np.append(timeseries[0], out)
    return out[:-1]


def timeseries(specific_search, duration, kg_to_tons):
    """
    Same meaning as analysis.py timeseries(...)

    specific_search:
        [(time, value), ...]
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
# SQLITE HELPERS
# =============================================================================
def get_duration(cur):
    row = cur.execute("SELECT Duration FROM Info").fetchone()
    if row is None:
        raise RuntimeError("Could not read Duration from Info table.")
    return int(row["Duration"])


def get_timestep_arrays(duration):
    timestep = np.arange(duration)
    half_length = int((len(timestep) + 1) / 12)
    new_timestep = timestep[:half_length]
    return timestep, new_timestep


def table_exists(cur, table_name):
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
    available = set(distinct_inbound_commodities(cur, prototype_name))
    for name in candidates:
        if name in available:
            return name

    raise RuntimeError(
        f'No matching inbound commodity found for "{prototype_name}". '
        f'Available inbound commodities: {sorted(available)}'
    )


def monthly_inflow_to_prototype(cur, duration, prototype_name, commodity_name):
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


def monthly_reprocessed_from_separationevents(cur, duration, prototype_name):
    """
    Direct notebook path:
    SELECT Time, Value FROM separationevents WHERE Type='UNF' AND Agentid=...
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


def reconstruct_lwr_reprocessing_throughput_from_outflows(cur, duration, prototype_name):
    """
    Fallback for LWR when separationevents is absent.

    Use TOTAL monthly outbound mass from uox_reprocessing,
    excluding feed-like commodities.

    This is the same idea that gave the working Figure 4.6 LWR result.
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

    print(f"[INFO] LWR fallback outbound commodities = {use_comms}")
    return total


# =============================================================================
# MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. checks
    # -------------------------------------------------------------------------
    if not os.path.exists(SQLITE_FILE):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_FILE}")

    if not os.path.exists("./bo/lwr_rep"):
        raise FileNotFoundError("Benchmark file not found: ./bo/lwr_rep")

    if not os.path.exists("./bo/sfr_rep"):
        raise FileNotFoundError("Benchmark file not found: ./bo/sfr_rep")

    # -------------------------------------------------------------------------
    # 2. open sqlite
    # -------------------------------------------------------------------------
    conn = sqlite3.connect(SQLITE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    duration = get_duration(cur)
    timestep, new_timestep = get_timestep_arrays(duration)
    has_separationevents = table_exists(cur, "separationevents")

    print(f"[INFO] duration = {duration}")
    print(f"[INFO] separationevents exists? {has_separationevents}")

    # -------------------------------------------------------------------------
    # 3. LWR annual reprocessing throughput
    # -------------------------------------------------------------------------
    if has_separationevents:
        # exact notebook logic
        uox_rep = monthly_reprocessed_from_separationevents(
            cur, duration, LWR_REPROC_PROTO
        )
        uox_rep = np.array(twosum(uox_rep), dtype=float)
        uox_rep = pull_in_one(uox_rep)
        print("[INFO] LWR throughput: using separationevents + twosum + pull_in_one")
    else:
        # minimal current-sqlite fallback
        uox_rep = reconstruct_lwr_reprocessing_throughput_from_outflows(
            cur=cur,
            duration=duration,
            prototype_name=LWR_REPROC_PROTO,
        )
        uox_rep = np.array(twosum(uox_rep), dtype=float)
        uox_rep = pull_in_one(uox_rep)
        print("[INFO] LWR throughput: using total outbound mass + twosum + pull_in_one")

    # -------------------------------------------------------------------------
    # 4. SFR annual reprocessing throughput
    # -------------------------------------------------------------------------
    if has_separationevents:
        # exact notebook logic
        sfr_rep = monthly_reprocessed_from_separationevents(
            cur, duration, SFR_REPROC_PROTO
        )
        sfr_rep = np.array(twosum(sfr_rep), dtype=float)
        print("[INFO] SFR throughput: using separationevents + twosum")
    else:
        # minimal fallback:
        # use feed entering sfr_reprocessing, because that matched notebook-style
        # SFR handling in Figure 4.6 and is the least speculative fallback here.
        sfr_reproc_feed = first_existing_inbound_commodity(
            cur,
            SFR_REPROC_PROTO,
            SFR_REPROC_IN_COMMOD_CANDIDATES,
        )
        sfr_rep = monthly_inflow_to_prototype(
            cur=cur,
            duration=duration,
            prototype_name=SFR_REPROC_PROTO,
            commodity_name=sfr_reproc_feed,
        )
        sfr_rep = np.array(twosum(sfr_rep), dtype=float)
        print(f"[INFO] SFR throughput: using inbound commodity {sfr_reproc_feed} + twosum")

    # -------------------------------------------------------------------------
    # 5. benchmark data
    # -------------------------------------------------------------------------
    data_uox_rep = np.array(read_from_data("./bo/lwr_rep"), dtype=float)
    data_sfr_rep = np.array(read_from_data("./bo/sfr_rep"), dtype=float)

    # -------------------------------------------------------------------------
    # 6. trim to common annual length
    # -------------------------------------------------------------------------
    n = min(
        len(new_timestep),
        len(uox_rep),
        len(sfr_rep),
        len(data_uox_rep),
        len(data_sfr_rep),
    )

    x = INIT_YEAR + new_timestep[:n]

    uox_rep = uox_rep[:n]
    sfr_rep = sfr_rep[:n]
    data_uox_rep = data_uox_rep[:n]
    data_sfr_rep = data_sfr_rep[:n]

    # -------------------------------------------------------------------------
    # 7. plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(x, uox_rep, label="LWR UNF [Cyclus]")
    plt.plot(x, data_uox_rep, label="LWR UNF [Feng et al.]", linestyle="-.")
    plt.plot(x, sfr_rep, label="SFR UNF [Cyclus]")
    plt.plot(x, data_sfr_rep, label="SFR UNF [Feng et al.]", linestyle="-.")

    plt.xlabel("Year")
    plt.ylabel("Annual Reprocessing Rate [t/y]")
    plt.title("Annual reprocessing throughputs")
    plt.legend()

    axes = plt.gca()
    # axes.autoscale(tight=True)    
    plt.grid(True)

    if SHOW_FIG:
        plt.show()

    conn.close()


if __name__ == "__main__":
    main()