#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_03_annual_fresh_fuel_loading.py

Reconstruct Figure 4.3 from Bae's notebook as faithfully as possible.

Target figure
-------------
Figure 4.3: Annual fresh fuel loading rates (first cores and reload fuel)

Important
---------
This script intentionally follows the notebook logic and its quirks
rather than trying to redesign the workflow.

Assumed working directory
-------------------------
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

    Despite the name, this sums 12 monthly values after removing the first entry.
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
    - truncate the last 4 entries
    """
    with open(data_path, "r") as file:
        timeseries = [0, 0]
        for row in file:
            value = float(row.replace("\n", ""))
            timeseries.append(value)

    return timeseries[:-4]


# =============================================================================
# SQLITE / CYCLUS HELPERS
# =============================================================================
def get_duration(cur):
    """
    Read simulation duration from Info table.
    """
    row = cur.execute("SELECT Duration FROM Info").fetchone()
    if row is None:
        raise RuntimeError("Could not read Duration from Info table.")
    return int(row["Duration"])


def get_timestep_arrays(duration):
    """
    Match notebook timestep handling.

    In the notebook:
        timestep = an.simulation_timesteps(cur)[3]
        half_length = int((len(timestep)+1) / 12)
        new_timestep = timestep[:half_length]

    Here we reconstruct the same behavior directly.
    """
    timestep = np.arange(duration)
    half_length = int((len(timestep) + 1) / 12)
    new_timestep = timestep[:half_length]
    return timestep, new_timestep


def get_agentids_for_prototype(cur, prototype_name):
    """
    Get all AgentIds whose prototype matches the requested prototype.
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
    Reconstruct the notebook's facility_commodity_flux(...) behavior
    for the specific use here: monthly mass flux time series by commodity.

    Parameters
    ----------
    cur : sqlite cursor
    duration : int
        Simulation duration in months
    agentids : list[int]
        Agent IDs belonging to one prototype (e.g. all lwr agents)
    commodities : list[str]
        Commodity names to track
    inbound : bool
        True  -> flux INTO these agents  (ReceiverId)
        False -> flux OUT OF these agents (SenderId)

    Returns
    -------
    flux_dict : dict
        Example:
            flux_dict["uox"] = monthly timeseries
    """
    flux_dict = {commod: np.zeros(duration, dtype=float) for commod in commodities}

    if len(agentids) == 0:
        return flux_dict

    # Build SQL placeholders
    agent_placeholders = ",".join(["?"] * len(agentids))
    commod_placeholders = ",".join(["?"] * len(commodities))

    # ReceiverId for inbound to facility, SenderId for outbound from facility
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
        q = float(row["quantity"]) / 1000.0

        if 0 <= t < duration and c in flux_dict:
            flux_dict[c][t] += q

    return flux_dict


# =============================================================================
# MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. Basic file checks
    # -------------------------------------------------------------------------
    if not os.path.exists(SQLITE_FILE):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_FILE}")

    if not os.path.exists("./bo/lwr_fuel_loaded"):
        raise FileNotFoundError("Benchmark file not found: ./bo/lwr_fuel_loaded")

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
    # 3. LWR fresh fuel loading
    #    Notebook:
    #       lwr_agentid = an.prototype_id(cur, 'lwr')
    #       lwr_load = an.facility_commodity_flux(cur, lwr_agentid, ['uox'], False, False)
    #
    # Here we reproduce the intended meaning for this figure:
    # fuel flowing INTO lwr agents with commodity 'uox'
    # -------------------------------------------------------------------------
    lwr_agentids = get_agentids_for_prototype(cur, "lwr")
    lwr_load = build_monthly_flux_timeseries(
        cur,
        duration,
        lwr_agentids,
        ["uox"],
        inbound=True
    )

    # Notebook boundary-condition fix
    # The first major point corresponds to the initial loading of all LWRs
    # and is handled manually there.
    if len(lwr_load["uox"]) > 2:
        lwr_load["uox"][1] = lwr_load["uox"][2]
        lwr_load["uox"][0] = lwr_load["uox"][1]

    new_lwr_load = twosum(lwr_load["uox"])

    if len(new_lwr_load) > 2:
        new_lwr_load[1] = new_lwr_load[2]

    # -------------------------------------------------------------------------
    # 4. SFR fresh fuel loading
    #    Notebook:
    #       sfr_agentid = an.prototype_id(cur, 'sfr')
    #       sfr_load = an.facility_commodity_flux(cur, sfr_agentid, ['sfr_fuel'], False, False)
    # -------------------------------------------------------------------------
    sfr_agentids = get_agentids_for_prototype(cur, "sfr")
    sfr_load = build_monthly_flux_timeseries(
        cur,
        duration,
        sfr_agentids,
        ["sfr_fuel"],
        inbound=True
    )

    new_sfr_load = twosum(sfr_load["sfr_fuel"])

    # -------------------------------------------------------------------------
    # 5. Benchmark/reference data
    # -------------------------------------------------------------------------
    data_lwr_load = read_from_data("./bo/lwr_fuel_loaded")
    data_sfr_load = read_from_data("./bo/sfr_fuel_loaded")

    # -------------------------------------------------------------------------
    # 6. Trim all arrays to common plotting length
    # -------------------------------------------------------------------------
    n = min(
        len(new_timestep),
        len(new_lwr_load),
        len(new_sfr_load),
        len(data_lwr_load),
        len(data_sfr_load),
    )

    x = INIT_YEAR + new_timestep[:n]

    new_lwr_load = np.array(new_lwr_load[:n], dtype=float)
    new_sfr_load = np.array(new_sfr_load[:n], dtype=float)
    data_lwr_load = np.array(data_lwr_load[:n], dtype=float)
    data_sfr_load = np.array(data_sfr_load[:n], dtype=float)

    # -------------------------------------------------------------------------
    # 7. Plot
    #    Notebook plotting order is preserved.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        new_lwr_load,
        label="LWR Fresh Fuel [Cyclus]",
        alpha=0.7,
    )
    plt.plot(
        x,
        data_lwr_load,
        label="LWR Fresh Fuel [Feng et al.]",
        linestyle="-.",
    )

    plt.plot(
        x,
        new_sfr_load,
        label="SFR Fresh Fuel [Cyclus]",
        alpha=0.7,
    )
    plt.plot(
        x,
        data_sfr_load,
        label="SFR Fresh Fuel [Feng et al.]",
        linestyle="-.",
    )

    plt.xlabel("Year")
    plt.ylabel("Fresh Fuel Loading [MTHM]")
    plt.title("Fresh Fuel Loading")
    plt.legend()
    plt.grid(True)

    if SHOW_FIG:
        plt.show()

    # -------------------------------------------------------------------------
    # 8. Close database
    # -------------------------------------------------------------------------
    conn.close()


if __name__ == "__main__":
    main()