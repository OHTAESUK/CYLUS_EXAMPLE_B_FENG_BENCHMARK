#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_05_unf_in_mandatory_cooling_storage.py

Figure 4.5 reconstruction
Inventory of discharged UNF in mandatory cooling storage

Priority:
1. If the sqlite file contains the table 'storageinventory',
   follow the notebook logic directly.
2. If not, reconstruct the mandatory cooling inventory from
   inflow transactions into the storage facilities.

This makes the script more robust across Cyclus output variants.
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

# Cooling durations implied by the input file
# residence_time = 47  -> 48 months in cooling
# residence_time = 11  -> 12 months in cooling
LWR_COOLING_MONTHS = 48
SFR_COOLING_MONTHS = 12


# =============================================================================
# NOTEBOOK-LIKE HELPERS
# =============================================================================
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


# =============================================================================
# SQLITE HELPERS
# =============================================================================
def get_duration(cur):
    row = cur.execute("SELECT Duration FROM Info").fetchone()
    if row is None:
        raise RuntimeError("Could not read Duration from Info table.")
    return int(row["Duration"])


def get_timestep_arrays(duration):
    """
    Match notebook timestep handling.
    """
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


def get_agentids_for_prototype(cur, prototype_name):
    rows = cur.execute(
        """
        SELECT agentid
        FROM agententry
        WHERE prototype = ?
        """,
        (prototype_name,),
    ).fetchall()
    return [int(row["agentid"]) for row in rows]


def get_storage_cooling_timeseries_from_table(cur, prototype_name):
    """
    Direct notebook-style method:
        SELECT quantity FROM storageinventory
        WHERE prototype = ?
          AND status = "Cooling"

    Quantity is converted from kg to tHM.
    """
    rows = cur.execute(
        """
        SELECT quantity
        FROM storageinventory
        WHERE prototype = ?
          AND status = "Cooling"
        """,
        (prototype_name,),
    ).fetchall()

    timeseries = [0.0]
    for row in rows:
        timeseries.append(float(row["quantity"]) / 1000.0)   # kg -> tHM

    return np.array(timeseries, dtype=float)


def get_monthly_inflow_to_storage(cur, duration, prototype_name, in_commodity):
    """
    Reconstruct the monthly inflow into a storage facility from Transactions.

    Logic
    -----
    1. Find all AgentIds with the given prototype name
    2. Sum transaction quantities whose ReceiverId is one of those agents
       and whose commodity matches the storage inflow commodity

    Quantity is converted from kg to tHM.
    """
    agentids = get_agentids_for_prototype(cur, prototype_name)
    monthly = np.zeros(duration, dtype=float)

    if len(agentids) == 0:
        return monthly

    placeholders = ",".join(["?"] * len(agentids))

    query = f"""
        SELECT transactions.time AS time,
               resources.quantity AS quantity
        FROM transactions
        INNER JOIN resources
            ON transactions.resourceid = resources.resourceid
        WHERE transactions.receiverid IN ({placeholders})
          AND transactions.commodity = ?
    """

    params = list(agentids) + [in_commodity]
    rows = cur.execute(query, params).fetchall()

    for row in rows:
        t = int(row["time"])
        q = float(row["quantity"]) / 1000.0   # kg -> tHM
        if 0 <= t < duration:
            monthly[t] += q

    return monthly


def rolling_cooling_inventory_from_inflow(monthly_inflow, cooling_months):
    """
    Reconstruct the inventory currently under mandatory cooling.

    Interpretation
    --------------
    Material entering the storage remains in the "Cooling" state for a fixed
    number of months. Therefore, the cooling inventory at a given month is the
    rolling sum of the most recent 'cooling_months' inflows.

    Returns
    -------
    inventory : np.ndarray
        Monthly cooling inventory in tHM
    """
    inventory = np.zeros(len(monthly_inflow), dtype=float)

    for t in range(len(monthly_inflow)):
        t0 = max(0, t - cooling_months + 1)
        inventory[t] = np.sum(monthly_inflow[t0:t + 1])

    return inventory


def get_storage_cooling_timeseries(cur, duration, prototype_name, in_commodity, cooling_months):
    """
    Prefer notebook-direct storageinventory table when available.
    Otherwise reconstruct from Transactions.
    """
    if table_exists(cur, "storageinventory"):
        print(f"[INFO] Using storageinventory table for {prototype_name}")
        return get_storage_cooling_timeseries_from_table(cur, prototype_name)

    print(f"[INFO] storageinventory table not found.")
    print(f"[INFO] Reconstructing cooling inventory from Transactions for {prototype_name}")

    monthly_inflow = get_monthly_inflow_to_storage(
        cur=cur,
        duration=duration,
        prototype_name=prototype_name,
        in_commodity=in_commodity,
    )

    inventory = rolling_cooling_inventory_from_inflow(
        monthly_inflow=monthly_inflow,
        cooling_months=cooling_months,
    )

    return inventory


# =============================================================================
# MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. Basic file checks
    # -------------------------------------------------------------------------
    if not os.path.exists(SQLITE_FILE):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_FILE}")

    if not os.path.exists("./bo/lwr_cool"):
        raise FileNotFoundError("Benchmark file not found: ./bo/lwr_cool")

    if not os.path.exists("./bo/sfr_cool"):
        raise FileNotFoundError("Benchmark file not found: ./bo/sfr_cool")

    # -------------------------------------------------------------------------
    # 2. Open sqlite database
    # -------------------------------------------------------------------------
    conn = sqlite3.connect(SQLITE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    duration = get_duration(cur)
    timestep, new_timestep = get_timestep_arrays(duration)

    # -------------------------------------------------------------------------
    # 3. CYCLUS monthly cooling-storage inventory
    # -------------------------------------------------------------------------
    uox_storage_timeseries = get_storage_cooling_timeseries(
        cur=cur,
        duration=duration,
        prototype_name="uox_unf_storage",
        in_commodity="uox_unf",
        cooling_months=LWR_COOLING_MONTHS,
    )

    sfr_storage_timeseries = get_storage_cooling_timeseries(
        cur=cur,
        duration=duration,
        prototype_name="sfr_unf_storage",
        in_commodity="sfr_unf",
        cooling_months=SFR_COOLING_MONTHS,
    )

    # -------------------------------------------------------------------------
    # 4. Benchmark/reference annual data
    # -------------------------------------------------------------------------
    data_uox_storage = np.array(read_from_data("./bo/lwr_cool"), dtype=float)
    data_sfr_storage = np.array(read_from_data("./bo/sfr_cool"), dtype=float)

    # -------------------------------------------------------------------------
    # 5. Trim annual benchmark arrays to common length
    # -------------------------------------------------------------------------
    n_annual = min(len(new_timestep), len(data_uox_storage), len(data_sfr_storage))
    x_annual = INIT_YEAR + new_timestep[:n_annual]
    data_uox_storage = data_uox_storage[:n_annual]
    data_sfr_storage = data_sfr_storage[:n_annual]

    # Monthly x-axis for CYCLUS
    n_monthly_uox = min(len(timestep), len(uox_storage_timeseries))
    n_monthly_sfr = min(len(timestep), len(sfr_storage_timeseries))

    x_monthly_uox = INIT_YEAR + timestep[:n_monthly_uox] / 12.0
    x_monthly_sfr = INIT_YEAR + timestep[:n_monthly_sfr] / 12.0

    uox_storage_timeseries = uox_storage_timeseries[:n_monthly_uox]
    sfr_storage_timeseries = sfr_storage_timeseries[:n_monthly_sfr]

    # -------------------------------------------------------------------------
    # 6. Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        x_monthly_uox,
        uox_storage_timeseries,
        label="LWR UNF [Cyclus]",
    )
    plt.plot(
        x_annual,
        data_uox_storage,
        label="LWR UNF [Feng et al.]",
        linestyle="-.",
    )

    plt.plot(
        x_monthly_sfr,
        sfr_storage_timeseries,
        label="SFR UNF [Cyclus]",
    )
    plt.plot(
        x_annual,
        data_sfr_storage,
        label="SFR UNF [Feng et al.]",
        linestyle="-.",
    )

    plt.xlabel("Year")
    plt.ylabel("Used Fuel in Cooling Storage [tHM]")
    plt.title("Inventory of discharged UNF in mandatory cooling storage")
    plt.legend()

    axes = plt.gca()
    axes.autoscale(tight=True)
    plt.grid(True)

    if SHOW_FIG:
        plt.show()

    conn.close()


if __name__ == "__main__":
    main()