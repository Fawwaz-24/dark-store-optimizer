"""Small regression checks for the dark-store optimization core."""
import numpy as np
from optimizer import Economics, generate_candidates, generate_customers, optimize_problem


def test_distance_limit_and_cap_monotonicity():
    customers = generate_customers(1000)
    candidates = generate_candidates().iloc[:20].reset_index(drop=True)
    limit_km = 4.0
    result = optimize_problem(customers, candidates, limit_km, Economics(), 10, (2, 5, 10, 20, 25))

    served = result["assignments"].query("service_status == 'Served'")
    assert (served["assigned_distance_km"] <= limit_km + 1e-8).all()
    assert len(result["selected_indices"]) <= 10
    profits = result["scenario_df"]["Daily profit (₹)"].to_numpy()
    assert np.all(np.diff(profits) >= -1e-6)
    assert abs(result["kpis"]["Daily profit (₹)"] - (
        result["kpis"]["Customer contribution/day (₹)"] - result["kpis"]["Store operating cost/day (₹)"]
    )) < 1e-6


if __name__ == "__main__":
    test_distance_limit_and_cap_monotonicity()
    print("All optimizer regression checks passed.")
