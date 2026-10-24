"""Core location-allocation heuristic for the Chennai dark-store optimizer.

Distances are great-circle (Haversine) distances in kilometres. Customer and
candidate coordinates are synthetic, reproducibly generated around Chennai.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# Approximate centre points of Chennai neighbourhoods; customers and candidate
# sites are jittered around these anchors, so this is illustrative rather than
# a survey of actual addresses.
NEIGHBOURHOODS = [
    ("Avadi", 13.1147, 80.1018),
    ("Ambattur", 13.1143, 80.1548),
    ("Mogappair", 13.0830, 80.1760),
    ("Anna Nagar", 13.0850, 80.2101),
    ("Koyambedu", 13.0694, 80.1948),
    ("Vadapalani", 13.0500, 80.2121),
    ("Nungambakkam", 13.0569, 80.2425),
    ("Kilpauk", 13.0820, 80.2430),
    ("Egmore", 13.0780, 80.2612),
    ("Perambur", 13.1175, 80.2335),
    ("Tondiarpet", 13.1287, 80.2895),
    ("Royapuram", 13.1080, 80.2950),
    ("T. Nagar", 13.0418, 80.2337),
    ("Mylapore", 13.0368, 80.2676),
    ("Royapettah", 13.0569, 80.2683),
    ("Guindy", 13.0067, 80.2206),
    ("Adyar", 13.0067, 80.2570),
    ("Velachery", 12.9750, 80.2200),
    ("Thiruvanmiyur", 12.9830, 80.2590),
    ("Perungudi", 12.9612, 80.2440),
    ("Thoraipakkam", 12.9368, 80.2364),
    ("Sholinganallur", 12.9010, 80.2279),
    ("Medavakkam", 12.9207, 80.1920),
    ("Pallavaram", 12.9675, 80.1491),
    ("Chromepet", 12.9516, 80.1462),
    ("Tambaram", 12.9249, 80.1000),
    ("Porur", 13.0351, 80.1562),
    ("Poonamallee", 13.0483, 80.1104),
    ("Kundrathur", 12.9990, 80.0970),
    ("Siruseri", 12.8301, 80.2205),
    ("Kelambakkam", 12.7879, 80.2206),
    ("Puzhal", 13.1630, 80.2040),
]

# Customer distribution weights are illustrative, not official population data.
CUSTOMER_HUBS = [
    ("Anna Nagar", 13.0850, 80.2101, 0.075),
    ("Ambattur", 13.1143, 80.1548, 0.045),
    ("Avadi", 13.1147, 80.1018, 0.025),
    ("Koyambedu", 13.0694, 80.1948, 0.045),
    ("Vadapalani", 13.0500, 80.2121, 0.055),
    ("T. Nagar", 13.0418, 80.2337, 0.085),
    ("Nungambakkam", 13.0569, 80.2425, 0.045),
    ("Egmore", 13.0780, 80.2612, 0.040),
    ("Perambur", 13.1175, 80.2335, 0.035),
    ("Tondiarpet", 13.1287, 80.2895, 0.025),
    ("Mylapore", 13.0368, 80.2676, 0.050),
    ("Guindy", 13.0067, 80.2206, 0.055),
    ("Adyar", 13.0067, 80.2570, 0.060),
    ("Velachery", 12.9750, 80.2200, 0.065),
    ("Thiruvanmiyur", 12.9830, 80.2590, 0.045),
    ("Perungudi", 12.9612, 80.2440, 0.045),
    ("Thoraipakkam", 12.9368, 80.2364, 0.040),
    ("Sholinganallur", 12.9010, 80.2279, 0.040),
    ("Medavakkam", 12.9207, 80.1920, 0.040),
    ("Pallavaram", 12.9675, 80.1491, 0.035),
    ("Chromepet", 12.9516, 80.1462, 0.030),
    ("Tambaram", 12.9249, 80.1000, 0.035),
]

@dataclass(frozen=True)
class Economics:
    orders_per_customer_day: float = 0.18
    contribution_per_order: float = 120.0
    delivery_cost_per_km_per_order: float = 5.0
    store_operating_cost_per_day: float = 3500.0


def haversine_matrix(customer_lat: np.ndarray, customer_lon: np.ndarray,
                     store_lat: np.ndarray, store_lon: np.ndarray) -> np.ndarray:
    """Return an N-customer x M-store great-circle distance matrix, in km."""
    lat1 = np.radians(np.asarray(customer_lat, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(customer_lon, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(store_lat, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(store_lon, dtype=float))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(a))


def generate_candidates() -> pd.DataFrame:
    """Create a stable set of 50 plausible candidate coordinates around Chennai."""
    rng = np.random.default_rng(9147)
    rows = []
    # Repeat each anchor with two nearby alternatives, representing distinct sites.
    for i in range(50):
        name, lat, lon = NEIGHBOURHOODS[(i * 13) % len(NEIGHBOURHOODS)]
        alt = "A" if i // len(NEIGHBOURHOODS) == 0 else "B"
        # Small site-level perturbation: approximately a few hundred metres.
        lat_jitter = rng.normal(0.0, 0.0016)
        lon_jitter = rng.normal(0.0, 0.0019)
        rows.append({
            "store_id": f"DS{i+1:02d}",
            "store_name": f"{name} · {alt}{(i // (2 * len(NEIGHBOURHOODS))) + 1 if i >= 2 * len(NEIGHBOURHOODS) else ''}".strip(),
            "latitude": lat + lat_jitter,
            "longitude": lon + lon_jitter,
        })
    # Ensure every candidate label is unique even where anchors repeat.
    for i, row in enumerate(rows):
        row["store_name"] = f"{row['store_name']} (Option {i+1})"
    return pd.DataFrame(rows)


def generate_customers(n: int, seed: int = 20261024) -> pd.DataFrame:
    """Generate reproducible synthetic customer coordinates across Chennai."""
    rng = np.random.default_rng(seed)
    hubs = np.array([[h[1], h[2]] for h in CUSTOMER_HUBS], dtype=float)
    weights = np.array([h[3] for h in CUSTOMER_HUBS], dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(len(hubs), size=int(n), p=weights)
    # Neighbourhood-level spread, with modest variation in each selected hub.
    lat_sigma_km = rng.uniform(0.45, 1.25, size=int(n))
    lon_sigma_km = rng.uniform(0.45, 1.35, size=int(n))
    lat = hubs[chosen, 0] + rng.normal(0, 1, int(n)) * lat_sigma_km / 111.0
    lon_scale = 111.0 * np.cos(np.radians(hubs[chosen, 0]))
    lon = hubs[chosen, 1] + rng.normal(0, 1, int(n)) * lon_sigma_km / lon_scale
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, int(n) + 1)],
        "latitude": lat,
        "longitude": lon,
        "neighbourhood_seed": [CUSTOMER_HUBS[j][0] for j in chosen],
    })


def make_contribution_matrix(distances_km: np.ndarray, max_distance_km: float,
                             economics: Economics) -> np.ndarray:
    """Expected daily contribution per customer-store pair; 0 means not assigned."""
    per_order_margin = (
        economics.contribution_per_order
        - economics.delivery_cost_per_km_per_order * distances_km
    )
    contribution = economics.orders_per_customer_day * per_order_margin
    contribution = np.where(distances_km <= float(max_distance_km), contribution, 0.0)
    # A customer never needs to be served at a negative contribution.
    return np.maximum(contribution, 0.0).astype(np.float64, copy=False)


def _best_values(scores: np.ndarray, selected: list[int]) -> np.ndarray:
    if not selected:
        return np.zeros(scores.shape[0], dtype=float)
    return np.max(scores[:, selected], axis=1)


def _objective(scores: np.ndarray, selected: list[int], fixed_cost: float) -> float:
    return float(np.sum(_best_values(scores, selected)) - fixed_cost * len(selected))


def _improve_selection(scores: np.ndarray, selected: list[int], fixed_cost: float,
                       cap: int, max_steps: int = 300) -> list[int]:
    """Greedy additions followed by profitable removals and one-store swaps.

    This is a deterministic local-search heuristic for the discrete facility
    location problem. Every accepted move strictly increases the objective.
    """
    selected = list(dict.fromkeys(selected))
    m = scores.shape[1]
    eps = 1e-7
    for _ in range(max_steps):
        current = _best_values(scores, selected)
        current_obj = float(current.sum() - fixed_cost * len(selected))

        # 1) Remove any redundant store if doing so improves net profit.
        best_remove_gain = eps
        remove_position = None
        for pos in range(len(selected)):
            remainder = selected[:pos] + selected[pos + 1:]
            without = _best_values(scores, remainder)
            gain = float(np.sum(without - current) + fixed_cost)
            if gain > best_remove_gain:
                best_remove_gain, remove_position = gain, pos
        if remove_position is not None:
            selected.pop(remove_position)
            continue

        # 2) Add the candidate with the highest positive incremental net profit.
        if len(selected) < cap:
            best_add_gain = eps
            best_add = None
            for j in range(m):
                if j in selected:
                    continue
                gain = float(np.maximum(current, scores[:, j]).sum() - current.sum() - fixed_cost)
                if gain > best_add_gain:
                    best_add_gain, best_add = gain, j
            if best_add is not None:
                selected.append(best_add)
                continue

        # 3) Try all one-for-one swaps. Opening cost cancels for a swap.
        if selected and len(selected) < m:
            best_swap_gain = eps
            best_swap = None
            selected_set = set(selected)
            for pos in range(len(selected)):
                remainder = selected[:pos] + selected[pos + 1:]
                base = _best_values(scores, remainder)
                for j in range(m):
                    if j in selected_set:
                        continue
                    gain = float(np.maximum(base, scores[:, j]).sum() - current.sum())
                    if gain > best_swap_gain:
                        best_swap_gain, best_swap = gain, (pos, j)
            if best_swap is not None:
                pos, j = best_swap
                selected[pos] = j
                continue

        # No improving move remains within the selected neighbourhood.
        break

    # Defensive feasibility check.
    if len(selected) > cap or len(set(selected)) != len(selected):
        raise RuntimeError("The optimizer returned an invalid facility selection.")
    return selected


def solve_caps(scores: np.ndarray, fixed_cost: float, caps: Iterable[int]) -> dict[int, list[int]]:
    """Solve caps in ascending order, warm-starting each scenario from the last.

    Because every local-search move must improve profit, the reported profit
    cannot fall when the permitted number of stores increases.
    """
    snapshots: dict[int, list[int]] = {}
    selected: list[int] = []
    for cap in sorted(set(max(0, min(int(c), scores.shape[1])) for c in caps)):
        selected = _improve_selection(scores, selected, fixed_cost, cap)
        snapshots[cap] = selected.copy()
    return snapshots


def assignment_for_selection(customers: pd.DataFrame, candidates: pd.DataFrame,
                             distances_km: np.ndarray, scores: np.ndarray,
                             selected: list[int], economics: Economics,
                             fixed_cost: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build customer assignments, per-store metrics, and overall KPI values."""
    n = len(customers)
    if selected:
        selected_scores = scores[:, selected]
        local_pos = np.argmax(selected_scores, axis=1)
        best_score = selected_scores[np.arange(n), local_pos]
        assigned_idx = np.asarray(selected, dtype=int)[local_pos]
        served_mask = best_score > 0
        assigned_idx = np.where(served_mask, assigned_idx, -1)
        assigned_distance = np.full(n, np.nan, dtype=float)
        assigned_distance[served_mask] = distances_km[np.arange(n)[served_mask], assigned_idx[served_mask]]
        assigned_contribution = np.where(served_mask, best_score, 0.0)
    else:
        assigned_idx = np.full(n, -1, dtype=int)
        assigned_distance = np.full(n, np.nan, dtype=float)
        assigned_contribution = np.zeros(n, dtype=float)
        served_mask = np.zeros(n, dtype=bool)

    assigned_names = np.full(n, "Unserved", dtype=object)
    assigned_ids = np.full(n, "", dtype=object)
    if selected:
        for idx in selected:
            mask = assigned_idx == idx
            assigned_names[mask] = candidates.iloc[idx]["store_name"]
            assigned_ids[mask] = candidates.iloc[idx]["store_id"]

    assignments = customers.copy()
    assignments["assigned_store_id"] = assigned_ids
    assignments["assigned_store"] = assigned_names
    assignments["assigned_distance_km"] = assigned_distance
    assignments["expected_contribution_inr_day"] = assigned_contribution
    assignments["service_status"] = np.where(served_mask, "Served", "Unserved")

    store_rows = []
    selected_set = set(selected)
    for idx in selected:
        mask = assigned_idx == idx
        count = int(mask.sum())
        contribution = float(assigned_contribution[mask].sum())
        store_rows.append({
            "Store ID": candidates.iloc[idx]["store_id"],
            "Candidate location": candidates.iloc[idx]["store_name"],
            "Latitude": candidates.iloc[idx]["latitude"],
            "Longitude": candidates.iloc[idx]["longitude"],
            "Customers assigned": count,
            "Expected orders/day": count * economics.orders_per_customer_day,
            "Customer contribution/day (₹)": contribution,
            "Operating cost/day (₹)": fixed_cost,
            "Store profit contribution/day (₹)": contribution - fixed_cost,
            "Mean delivery distance (km)": float(np.nanmean(assigned_distance[mask])) if count else np.nan,
            "Max delivery distance (km)": float(np.nanmax(assigned_distance[mask])) if count else np.nan,
        })
    stores = pd.DataFrame(store_rows)
    served_n = int(served_mask.sum())
    customer_contribution = float(assigned_contribution.sum())
    store_cost = float(fixed_cost * len(selected))
    profit = customer_contribution - store_cost
    avg_distance = float(np.nanmean(assigned_distance)) if served_n else 0.0
    max_assigned_distance = float(np.nanmax(assigned_distance)) if served_n else 0.0
    expected_orders = served_n * economics.orders_per_customer_day
    kpis = {
        "Customers": n,
        "Candidate locations": len(candidates),
        "Stores opened": len(selected_set),
        "Customers served": served_n,
        "Customers unserved": n - served_n,
        "Customer coverage (%)": 100.0 * served_n / n if n else 0.0,
        "Expected orders/day": expected_orders,
        "Mean assigned distance (km)": avg_distance,
        "Maximum assigned distance (km)": max_assigned_distance,
        "Customer contribution/day (₹)": customer_contribution,
        "Store operating cost/day (₹)": store_cost,
        "Daily profit (₹)": profit,
    }
    return assignments, stores, kpis


def optimize_problem(customers: pd.DataFrame, candidates: pd.DataFrame,
                     max_distance_km: float, economics: Economics,
                     max_store_cap: int, scenario_caps: Iterable[int] = (2, 5, 10, 20, 25)) -> dict:
    """Construct distances/scores and optimize scenario caps plus detail cap."""
    n_candidates = len(candidates)
    max_store_cap = max(0, min(int(max_store_cap), n_candidates))
    distances = haversine_matrix(
        customers["latitude"].to_numpy(), customers["longitude"].to_numpy(),
        candidates["latitude"].to_numpy(), candidates["longitude"].to_numpy(),
    )
    scores = make_contribution_matrix(distances, max_distance_km, economics)
    requested_caps = [max(0, min(int(cap), n_candidates)) for cap in scenario_caps]
    requested_caps.append(max_store_cap)
    snapshots = solve_caps(scores, economics.store_operating_cost_per_day, requested_caps)
    selected = snapshots[max_store_cap]
    assignments, store_metrics, kpis = assignment_for_selection(
        customers, candidates, distances, scores, selected, economics,
        economics.store_operating_cost_per_day,
    )

    # Scenario curve always uses the standard assignment caps requested.
    scenario_rows = []
    for original_cap in scenario_caps:
        effective_cap = max(0, min(int(original_cap), n_candidates))
        scenario_selected = snapshots[effective_cap]
        scenario_obj = _objective(scores, scenario_selected, economics.store_operating_cost_per_day)
        if scenario_selected:
            scenario_best = _best_values(scores, scenario_selected)
            scenario_served = int((scenario_best > 0).sum())
        else:
            scenario_served = 0
        scenario_rows.append({
            "Store cap": int(original_cap),
            "Effective cap": effective_cap,
            "Stores opened": len(scenario_selected),
            "Customers served": scenario_served,
            "Daily profit (₹)": scenario_obj,
        })
    scenario_df = pd.DataFrame(scenario_rows)
    return {
        "distances_km": distances,
        "scores": scores,
        "selected_indices": selected,
        "snapshots": snapshots,
        "assignments": assignments,
        "store_metrics": store_metrics,
        "kpis": kpis,
        "scenario_df": scenario_df,
        "economics": economics,
    }
