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


def _capacity_assignment(scores: np.ndarray, selected: list[int],
                         max_customers_per_store: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign customers to selected stores without exceeding store capacity.

    Each customer is indivisible and has the same expected order rate. The
    capacity in orders/day is therefore converted to a maximum number of
    customers per store. If unconstrained nearest-best assignments are already
    within capacity, they are retained. Otherwise a deterministic greedy
    allocation prioritizes customers with the largest best available
    contribution, then tries their remaining stores in descending score order.
    """
    n = scores.shape[0]
    assigned_store = np.full(n, -1, dtype=int)
    assigned_score = np.zeros(n, dtype=float)
    if not selected or max_customers_per_store <= 0:
        return assigned_store, assigned_score

    selected_scores = scores[:, selected]
    n_selected = len(selected)
    local_best = np.argmax(selected_scores, axis=1)
    best_score = selected_scores[np.arange(n), local_best]
    served = best_score > 0
    loads = np.bincount(local_best[served], minlength=n_selected)

    # Fast path: no capacity is exceeded by the unconstrained assignment.
    if np.all(loads <= max_customers_per_store):
        assigned_store[served] = np.asarray(selected, dtype=int)[local_best[served]]
        assigned_score[served] = best_score[served]
        return assigned_store, assigned_score

    # Capacity binds. Assign high-value customers first and offer each customer
    # their best remaining selected store with spare capacity.
    preference_order = np.argsort(-selected_scores, axis=1, kind="stable")
    customer_priority = np.argsort(-best_score, kind="stable")
    loads = np.zeros(n_selected, dtype=int)
    selected_array = np.asarray(selected, dtype=int)
    for i in customer_priority:
        if best_score[i] <= 0:
            break
        for local_j in preference_order[i]:
            value = selected_scores[i, local_j]
            if value <= 0:
                break
            if loads[local_j] < max_customers_per_store:
                assigned_store[i] = selected_array[local_j]
                assigned_score[i] = value
                loads[local_j] += 1
                break
    return assigned_store, assigned_score


def _capacity_objective(scores: np.ndarray, selected: list[int], fixed_cost: float,
                        max_customers_per_store: int) -> float:
    _, contributions = _capacity_assignment(scores, selected, max_customers_per_store)
    return float(contributions.sum() - fixed_cost * len(selected))


def _improve_selection(scores: np.ndarray, selected: list[int], fixed_cost: float,
                       cap: int, max_customers_per_store: int,
                       max_steps: int = 300) -> list[int]:
    """Greedy capacitated facility selection with profitable store removals.

    Every accepted addition or removal is evaluated with capacity-constrained
    customer assignment and strictly increases the objective.
    """
    selected = list(dict.fromkeys(selected))
    m = scores.shape[1]
    eps = 1e-7
    objective_cache: dict[tuple[int, ...], float] = {}

    def obj(indices: list[int]) -> float:
        key = tuple(sorted(indices))
        if key not in objective_cache:
            objective_cache[key] = _capacity_objective(scores, list(key), fixed_cost,
                                                       max_customers_per_store)
        return objective_cache[key]

    for _ in range(max_steps):
        current_obj = obj(selected)

        # First remove a redundant store if that improves capacity-aware profit.
        best_remove_gain = eps
        remove_position = None
        for pos in range(len(selected)):
            remainder = selected[:pos] + selected[pos + 1:]
            gain = obj(remainder) - current_obj
            if gain > best_remove_gain:
                best_remove_gain, remove_position = gain, pos
        if remove_position is not None:
            selected.pop(remove_position)
            continue

        # Add the candidate yielding the largest positive capacity-aware gain.
        if len(selected) < cap:
            best_add_gain = eps
            best_add = None
            selected_set = set(selected)
            for j in range(m):
                if j in selected_set:
                    continue
                gain = obj(selected + [j]) - current_obj
                if gain > best_add_gain:
                    best_add_gain, best_add = gain, j
            if best_add is not None:
                selected.append(best_add)
                continue
        break

    if len(selected) > cap or len(set(selected)) != len(selected):
        raise RuntimeError("The optimizer returned an invalid facility selection.")
    return selected


def solve_caps(scores: np.ndarray, fixed_cost: float, caps: Iterable[int],
               max_customers_per_store: int) -> dict[int, list[int]]:
    """Solve caps in ascending order, warm-starting from the prior scenario."""
    snapshots: dict[int, list[int]] = {}
    selected: list[int] = []
    for cap in sorted(set(max(0, min(int(c), scores.shape[1])) for c in caps)):
        selected = _improve_selection(scores, selected, fixed_cost, cap,
                                      max_customers_per_store)
        snapshots[cap] = selected.copy()
    return snapshots


def assignment_for_selection(customers: pd.DataFrame, candidates: pd.DataFrame,
                             distances_km: np.ndarray, scores: np.ndarray,
                             selected: list[int], economics: Economics,
                             fixed_cost: float,
                             max_orders_per_store_day: float = 300.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build capacity-feasible customer assignments, store metrics and KPIs."""
    n = len(customers)
    max_customers_per_store = int(np.floor(
        max_orders_per_store_day / economics.orders_per_customer_day + 1e-12
    ))
    assigned_idx, assigned_contribution = _capacity_assignment(
        scores, selected, max_customers_per_store
    )
    served_mask = assigned_idx >= 0
    assigned_distance = np.full(n, np.nan, dtype=float)
    if served_mask.any():
        assigned_distance[served_mask] = distances_km[
            np.arange(n)[served_mask], assigned_idx[served_mask]
        ]

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
            "Order capacity/day": float(max_orders_per_store_day),
            "Capacity utilization (%)": (
                100.0 * count * economics.orders_per_customer_day / max_orders_per_store_day
                if max_orders_per_store_day > 0 else 0.0
            ),
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
        "Store capacity (orders/day)": float(max_orders_per_store_day),
        "Total open-store capacity (orders/day)": float(max_orders_per_store_day * len(selected_set)),
        "Maximum store capacity utilization (%)": (
            float(stores["Capacity utilization (%)"].max()) if not stores.empty else 0.0
        ),
        "Mean assigned distance (km)": avg_distance,
        "Maximum assigned distance (km)": max_assigned_distance,
        "Customer contribution/day (₹)": customer_contribution,
        "Store operating cost/day (₹)": store_cost,
        "Daily profit (₹)": profit,
    }
    return assignments, stores, kpis


def optimize_problem(customers: pd.DataFrame, candidates: pd.DataFrame,
                     max_distance_km: float, economics: Economics,
                     max_store_cap: int, scenario_caps: Iterable[int] = (2, 5, 10, 20, 25),
                     max_orders_per_store_day: float = 300.0) -> dict:
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
    max_customers_per_store = int(np.floor(
        max_orders_per_store_day / economics.orders_per_customer_day + 1e-12
    ))
    snapshots = solve_caps(scores, economics.store_operating_cost_per_day,
                           requested_caps, max_customers_per_store)
    selected = snapshots[max_store_cap]
    assignments, store_metrics, kpis = assignment_for_selection(
        customers, candidates, distances, scores, selected, economics,
        economics.store_operating_cost_per_day, max_orders_per_store_day,
    )

    # Scenario curve always uses the standard assignment caps requested.
    scenario_rows = []
    for original_cap in scenario_caps:
        effective_cap = max(0, min(int(original_cap), n_candidates))
        scenario_selected = snapshots[effective_cap]
        _, scenario_contributions = _capacity_assignment(
            scores, scenario_selected, max_customers_per_store
        )
        scenario_obj = float(
            scenario_contributions.sum()
            - economics.store_operating_cost_per_day * len(scenario_selected)
        )
        scenario_served = int((scenario_contributions > 0).sum())
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
        "max_orders_per_store_day": float(max_orders_per_store_day),
        "max_customers_per_store": max_customers_per_store,
    }
