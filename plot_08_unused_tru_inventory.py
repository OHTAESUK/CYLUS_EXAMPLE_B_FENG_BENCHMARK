#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_08_unused_tru_inventory.py

Figure 4.8 reconstruction
Inventory of unused TRU recovered from UNF

This version follows the notebook logic directly, now that the sqlite
DOES contain SeparationEvents.

Notebook-equivalent logic
-------------------------
1) separated Pu timeseries:
       uox_pu + sfr_pu from SeparationEvents
2) Pu usage:
       sfr_mixer_sfr outflux of sfr_fuel from analysis.facility_commodity_flux(...)
       multiplied by 0.1387
3) leftover monthly inventory:
       leftover[t] = leftover[t-1] + pu_inv_timeseries[t] - pu_used[t]
4) final annual CYCLUS curve:
       find_min(12, leftover)
5) benchmark:
       ./bo/surplus_tru

No automatic figure saving.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import analysis as an


# =============================================================================
# USER INPUT
# =============================================================================
SQLITE_FILE = "output_TsOh.sqlite"
SHOW_FIG = True


# =============================================================================
# NOTEBOOK HELPERS
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


def find_min(group_by, timeseries):
    """
    Exact notebook helper used for Figure 4.8 plotting.

    Important:
    ----------
    This does NOT compute annual sums.
    It takes the MINIMUM value in each grouped block.

    The notebook uses:
        group = timeseries[i * group_by : i * group_by + group_by -1]
        add = np.min(group)
    """
    result = []
    for i in range(0, int(np.floor(len(timeseries) / group_by))):
        group = timeseries[i * group_by : i * group_by + group_by - 1]
        add = np.min(group)
        result.append(add)
    return np.array(result, dtype=float)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. Basic checks
    # -------------------------------------------------------------------------
    if not os.path.exists(SQLITE_FILE):
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_FILE}")

    if not os.path.exists("./bo/surplus_tru"):
        raise FileNotFoundError("Benchmark file not found: ./bo/surplus_tru")

    # -------------------------------------------------------------------------
    # 2. Notebook-style cursor and timestep info
    # -------------------------------------------------------------------------
    cur = an.cursor(SQLITE_FILE)
    init_year, init_month, duration, timestep = an.simulation_timesteps(cur)
    new_timestep = timestep[: int((len(timestep) + 1) / 12)]

    print(f"[INFO] duration = {duration}")

    # -------------------------------------------------------------------------
    # 3. Separated Pu timeseries from SeparationEvents
    # -------------------------------------------------------------------------
    # The notebook explicitly builds:
    #   pu_inv_timeseries = timeseries(uox_pu) + timeseries(sfr_pu)
    #
    # We preserve that exact structure here.
    influx1 = cur.execute(
        'SELECT time, Value FROM SeparationEvents '
        'WHERE type = "uox_pu"'
    ).fetchall()

    influx2 = cur.execute(
        'SELECT time, Value FROM SeparationEvents '
        'WHERE type = "sfr_pu"'
    ).fetchall()

    pu_inv_timeseries = (
        np.array(an.timeseries(influx1, duration, True), dtype=float)
        + np.array(an.timeseries(influx2, duration, True), dtype=float)
    )

    print(f"[INFO] nonzero months in recovered Pu = {np.count_nonzero(pu_inv_timeseries)}")

    # -------------------------------------------------------------------------
    # 4. Pu usage from sfr_mixer_sfr
    # -------------------------------------------------------------------------
    # Notebook:
    #   sfr_agentid = an.prototype_id(cur, 'sfr_mixer_sfr')
    #   sfr_load = an.facility_commodity_flux(cur, sfr_agentid, ['sfr_fuel'], True, False)
    #   pu_used = np.array(sfr_load['sfr_fuel']) * 0.1387
    sfr_agentid = an.prototype_id(cur, 'sfr_mixer_sfr')

    sfr_load = an.facility_commodity_flux(
        cur,
        sfr_agentid,
        ['sfr_fuel'],
        True,   # exact notebook: outflux
        False   # exact notebook: non-cumulative request at helper call level
    )

    pu_used = np.array(sfr_load['sfr_fuel'], dtype=float) * 0.1387

    # notebook one-step forward shift
    pu_used = np.append(np.array(pu_used, dtype=float), pu_used[-1])
    pu_used = pu_used[1:]

    print(f"[INFO] nonzero months in Pu used = {np.count_nonzero(pu_used)}")

    # -------------------------------------------------------------------------
    # 5. Monthly leftover inventory
    # -------------------------------------------------------------------------
    # Keep the explicit notebook loop.
    leftover = np.zeros(len(pu_used), dtype=float)
    for indx, val in enumerate(leftover):
        if indx == 0:
            continue
        leftover[indx] = (
            leftover[indx - 1]
            + pu_inv_timeseries[indx]
            - pu_used[indx]
        )

    # -------------------------------------------------------------------------
    # 6. Final CYCLUS annual curve = exact notebook plotting helper
    # -------------------------------------------------------------------------
    cyclus_plot = find_min(12, leftover)

    # -------------------------------------------------------------------------
    # 7. Benchmark
    # -------------------------------------------------------------------------
    data_surplus_tru = read_from_data('./bo/surplus_tru')
    benchmark_plot = np.array(data_surplus_tru + [data_surplus_tru[-1]], dtype=float)

    # -------------------------------------------------------------------------
    # 8. Trim to common plotting length
    # -------------------------------------------------------------------------
    n = min(len(new_timestep), len(cyclus_plot), len(benchmark_plot))
    x = init_year + new_timestep[:n]

    # -------------------------------------------------------------------------
    # 9. Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(x, cyclus_plot[:n], label='CYCLUS')
    plt.plot(x, benchmark_plot[:n], label='Feng et al.', linestyle='-.')

    plt.xlabel('Year')
    plt.ylabel('Unused TRU from UNF [tHM]')
    plt.title('Inventory of unused TRU recovered from UNF')
    plt.legend()
    plt.grid(True)

    if SHOW_FIG:
        plt.show()


if __name__ == "__main__":
    main()