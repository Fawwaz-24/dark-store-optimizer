from __future__ import annotations

import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

from optimizer import Economics, generate_candidates, generate_customers, optimize_problem

st.set_page_config(
    page_title="Chennai Dark Store Optimizer",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
      .hero {padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(110deg,#102a43 0%,#0f766e 100%); color: white; margin-bottom: 1rem;}
      .hero h1 {color:white; margin:0; font-size:2rem;}
      .hero p {color:#e4f3f3; margin:.35rem 0 0 0; font-size:1rem;}
      .small-note {color:#627084; font-size:.87rem;}
      div[data-testid="stMetric"] {background:#fff; border:1px solid #e6ebf1; padding: .8rem 1rem; border-radius: 12px;}
      div[data-testid="stMetricLabel"] p {font-size:.85rem;}
      .section-label {font-weight:650; font-size:1.12rem; margin:.3rem 0 .25rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def rupees(value: float) -> str:
    return "₹" + f"{value:,.0f}"


@st.cache_data(show_spinner=False, max_entries=20)
def run_cached(n_customers: int, n_candidates: int, max_distance_km: float,
               orders_per_customer_day: float, contribution_per_order: float,
               delivery_cost_per_km_per_order: float, store_cost_per_day: float,
               max_store_cap: int, max_orders_per_store_day: float):
    customers = generate_customers(n_customers)
    candidates = generate_candidates().iloc[:n_candidates].reset_index(drop=True)
    economics = Economics(
        orders_per_customer_day=float(orders_per_customer_day),
        contribution_per_order=float(contribution_per_order),
        delivery_cost_per_km_per_order=float(delivery_cost_per_km_per_order),
        store_operating_cost_per_day=float(store_cost_per_day),
    )
    result = optimize_problem(
        customers=customers,
        candidates=candidates,
        max_distance_km=float(max_distance_km),
        economics=economics,
        max_store_cap=int(max_store_cap),
        scenario_caps=(2, 5, 10, 20, 25),
        max_orders_per_store_day=float(max_orders_per_store_day),
    )
    return customers, candidates, result


@st.cache_data(show_spinner=False, max_entries=20)
def run_assignment_sweep_cached(max_distance_km: float,
                                orders_per_customer_day: float,
                                contribution_per_order: float,
                                delivery_cost_per_km_per_order: float,
                                store_cost_per_day: float,
                                max_orders_per_store_day: float) -> pd.DataFrame:
    """Run the required fixed benchmark: 10,000 customers, 50 candidates,
    and maximum store caps of 2, 5, 10, 20 and 25.

    The active distance limit and economics are used, but customer/candidate
    samples are held fixed so the comparison is apples-to-apples.
    """
    customers = generate_customers(10000)
    candidates = generate_candidates().iloc[:50].reset_index(drop=True)
    economics = Economics(
        orders_per_customer_day=float(orders_per_customer_day),
        contribution_per_order=float(contribution_per_order),
        delivery_cost_per_km_per_order=float(delivery_cost_per_km_per_order),
        store_operating_cost_per_day=float(store_cost_per_day),
    )
    result = optimize_problem(
        customers=customers,
        candidates=candidates,
        max_distance_km=float(max_distance_km),
        economics=economics,
        max_store_cap=25,
        scenario_caps=(2, 5, 10, 20, 25),
        max_orders_per_store_day=float(max_orders_per_store_day),
    )
    return result["scenario_df"]


def make_solution_map(customers: pd.DataFrame, candidates: pd.DataFrame, result: dict,
                      max_distance_km: float) -> folium.Map:
    selected = result["selected_indices"]
    assignment = result["assignments"]
    store_metrics = result["store_metrics"]
    map_center = [13.035, 80.215]
    fmap = folium.Map(location=map_center, zoom_start=11, tiles="OpenStreetMap", control_scale=True)

    palette = [
        "#0f766e", "#2563eb", "#9333ea", "#ea580c", "#0891b2", "#be123c",
        "#4d7c0f", "#7c3aed", "#c2410c", "#0369a1", "#a21caf", "#15803d",
        "#b45309", "#1d4ed8", "#9f1239", "#0e7490", "#6d28d9", "#3f6212",
        "#c026d3", "#475569", "#a16207", "#047857", "#7e22ce", "#1e40af",
        "#c2410c",
    ]
    idx_to_color = {idx: palette[pos % len(palette)] for pos, idx in enumerate(selected)}
    store_info = (
        store_metrics.set_index("Store ID").to_dict("index")
        if not store_metrics.empty else {}
    )

    # Show the candidate pool distinctly from the open stores.
    candidate_layer = folium.FeatureGroup(name="All candidate sites", show=True)
    selected_set = set(selected)
    for idx, row in candidates.iterrows():
        is_open = idx in selected_set
        color = idx_to_color.get(idx, "#8b95a5") if is_open else "#8b95a5"
        label = "OPEN STORE" if is_open else "Candidate only"
        capacity_detail = ""
        if is_open and row["store_id"] in store_info:
            metric = store_info[row["store_id"]]
            capacity_detail = (
                f"<br>Expected orders/day: {metric['Expected orders/day']:.1f}"
                f"<br>Capacity/day: {metric['Order capacity/day']:.0f}"
                f"<br>Capacity utilization: {metric['Capacity utilization (%)']:.1f}%"
            )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8 if is_open else 4,
            color=color,
            weight=2 if is_open else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.95 if is_open else 0.55,
            popup=folium.Popup(
                f"<b>{row['store_id']}: {row['store_name']}</b><br>{label}<br>"
                f"Coordinates: {row['latitude']:.5f}, {row['longitude']:.5f}"
                f"{capacity_detail}",
                max_width=300,
            ),
            tooltip=f"{row['store_id']} · {row['store_name']} · {label}",
        ).add_to(candidate_layer)
        if is_open:
            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=max_distance_km * 1000,
                color=color,
                weight=1,
                fill=False,
                opacity=0.16,
                tooltip=f"{row['store_id']} service radius: {max_distance_km:g} km",
            ).add_to(candidate_layer)
    candidate_layer.add_to(fmap)

    # Customer markers are sampled for browser responsiveness; all customers
    # are included in the optimization and downloadable assignment data.
    customer_layer = folium.FeatureGroup(name="Customer sample", show=True)
    n = len(assignment)
    sample_n = min(1600, n)
    sample_idx = np.linspace(0, max(n - 1, 0), sample_n, dtype=int) if n else np.array([], dtype=int)
    for i in sample_idx:
        row = assignment.iloc[int(i)]
        if row["service_status"] == "Served":
            store_id = row["assigned_store_id"]
            try:
                candidate_idx = int(candidates.index[candidates["store_id"] == store_id][0])
            except (IndexError, ValueError):
                candidate_idx = selected[0] if selected else -1
            color = idx_to_color.get(candidate_idx, "#0f766e")
            popup_text = (
                f"<b>{row['customer_id']}</b><br>Served by {row['assigned_store']}<br>"
                f"Straight-line distance: {row['assigned_distance_km']:.2f} km"
            )
        else:
            color = "#cbd5e1"
            popup_text = f"<b>{row['customer_id']}</b><br>Unserved within the permitted range"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2.2,
            color=color,
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.72,
            popup=folium.Popup(popup_text, max_width=260),
        ).add_to(customer_layer)
    customer_layer.add_to(fmap)

    # Draw a small sample of real assignment links, not every line for 10,000 customers.
    if selected:
        served_idx = np.flatnonzero(assignment["service_status"].to_numpy() == "Served")
        if len(served_idx):
            rng = np.random.default_rng(733)
            line_idx = rng.choice(served_idx, size=min(90, len(served_idx)), replace=False)
            for i in line_idx:
                row = assignment.iloc[int(i)]
                store_matches = candidates.index[candidates["store_id"] == row["assigned_store_id"]]
                if len(store_matches) == 0:
                    continue
                store = candidates.iloc[int(store_matches[0])]
                store_idx = int(store_matches[0])
                folium.PolyLine(
                    locations=[[row["latitude"], row["longitude"]], [store["latitude"], store["longitude"]]],
                    color=idx_to_color.get(store_idx, "#0f766e"),
                    weight=1.1,
                    opacity=0.42,
                ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    legend_html = """
    <div style="position: fixed; bottom: 26px; left: 26px; z-index: 9999; background: white;
                padding: 10px 12px; border: 1px solid #ccd4dd; border-radius: 8px;
                font-size: 12px; box-shadow: 0 1px 6px #0002;">
      <b>Map legend</b><br>
      <span style="color:#0f766e;">●</span> Open dark store<br>
      <span style="color:#8b95a5;">●</span> Candidate not opened<br>
      <span style="color:#2563eb;">●</span> Customer marker colour = assigned store<br>
      <span style="color:#cbd5e1;">●</span> Unserved customer
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


def render_kpi_table(kpis: dict, max_distance_km: float, max_orders_per_store_day: float) -> pd.DataFrame:
    rows = [
        ("Customer count", f"{kpis['Customers']:,}"),
        ("Candidate dark-store sites", f"{kpis['Candidate locations']:,}"),
        ("Dark stores opened", f"{kpis['Stores opened']:,}"),
        ("Customers served", f"{kpis['Customers served']:,}"),
        ("Customers unserved", f"{kpis['Customers unserved']:,}"),
        ("Customer coverage", f"{kpis['Customer coverage (%)']:.1f}%"),
        ("Expected orders per day", f"{kpis['Expected orders/day']:,.1f}"),
        ("Capacity per dark store", f"{max_orders_per_store_day:,.0f} orders/day"),
        ("Total capacity of open stores", f"{kpis['Total open-store capacity (orders/day)']:,.0f} orders/day"),
        ("Maximum open-store capacity utilization", f"{kpis['Maximum store capacity utilization (%)']:.1f}%"),
        ("Capacity constraint", "SATISFIED" if kpis['Expected orders/day'] <= kpis['Total open-store capacity (orders/day)'] + 1e-8 and kpis['Maximum store capacity utilization (%)'] <= 100 + 1e-8 else "VIOLATION"),
        ("Mean customer–store distance", f"{kpis['Mean assigned distance (km)']:.2f} km"),
        ("Maximum assigned distance", f"{kpis['Maximum assigned distance (km)']:.2f} km"),
        ("Maximum permitted distance", f"{max_distance_km:.2f} km"),
        ("Distance constraint", "SATISFIED" if kpis['Maximum assigned distance (km)'] <= max_distance_km + 1e-8 else "VIOLATION"),
        ("Expected customer contribution / day", rupees(kpis["Customer contribution/day (₹)"])),
        ("Dark-store operating cost / day", rupees(kpis["Store operating cost/day (₹)"])),
        ("Net profit / day", rupees(kpis["Daily profit (₹)"])),
    ]
    return pd.DataFrame(rows, columns=["KPI", "Result"])


st.markdown(
    """
    <div class="hero">
      <h1>📍 Chennai Dark Store Location Optimizer</h1>
      <p>Choose candidate sites, enforce customer service distance and store order capacity, and maximize expected daily contribution after operating costs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Problem inputs")
    with st.form("optimization_inputs"):
        n_customers = st.number_input(
            "Number of customers", min_value=500, max_value=20000, value=10000, step=500,
            help="Synthetic customers are generated across Chennai using a fixed random seed.",
        )
        n_candidates = st.number_input(
            "Dark-store candidate locations", min_value=10, max_value=50, value=50, step=5,
            help="Choose from 50 illustrative candidate coordinates around Chennai.",
        )
        max_distance = st.slider(
            "Maximum customer–store distance (km)", min_value=1.0, max_value=15.0, value=5.0, step=0.5,
            help="A customer can only be assigned to an open store within this straight-line distance.",
        )
        max_store_cap = st.slider(
            "Maximum number of stores allowed to open", min_value=1, max_value=int(n_candidates), value=min(10, int(n_candidates)), step=1,
        )
        st.markdown("---")
        st.markdown("### Operational constraints")
        max_orders_per_store_day = st.number_input(
            "Maximum orders per dark store per day", min_value=50.0, max_value=5000.0,
            value=300.0, step=50.0,
            help="Hard throughput limit for EACH open store. Illustrative default: 300 orders/day. This is separate from customer demand.",
        )
        st.caption("Capacity is enforced separately for every open store.")
        st.markdown("### Economics assumptions")
        orders_per_customer_day = st.number_input(
            "Expected orders / customer / day", min_value=0.01, max_value=2.0, value=0.18, step=0.01,
            help="Customer demand assumption: 0.18 orders/day is approximately 5.4 orders per active customer per month.",
        )
        contribution_per_order = st.number_input(
            "Contribution per order before delivery (₹)", min_value=10.0, max_value=1000.0, value=120.0, step=10.0,
        )
        delivery_cost_per_km = st.number_input(
            "Delivery cost per order per km (₹)", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
        )
        store_cost_per_day = st.number_input(
            "Operating cost per store per day (₹)", min_value=0.0, max_value=100000.0, value=3500.0, step=250.0,
        )
        submit = st.form_submit_button("Run optimization & refresh KPIs/map", type="primary", use_container_width=True)

    st.markdown(
        "<p class='small-note'>All locations, customer demand and financial values are illustrative assumptions for the assignment—not live business data.</p>",
        unsafe_allow_html=True,
    )

params = {
    "n_customers": int(n_customers),
    "n_candidates": int(n_candidates),
    "max_distance": float(max_distance),
    "orders_per_customer_day": float(orders_per_customer_day),
    "contribution_per_order": float(contribution_per_order),
    "delivery_cost_per_km": float(delivery_cost_per_km),
    "store_cost_per_day": float(store_cost_per_day),
    "max_orders_per_store_day": float(max_orders_per_store_day),
    "max_store_cap": min(int(max_store_cap), int(n_candidates)),
}

if submit or "dark_store_result" not in st.session_state:
    # Each explicit run should refresh the results rather than risk a stale cached
    # object while a user is debugging deployment/input behaviour. The optimization
    # remains deterministic, so the same inputs return the same solution.
    if submit:
        run_cached.clear()
        run_assignment_sweep_cached.clear()
    with st.spinner("Generating Chennai demand points and optimizing dark-store locations…"):
        customers, candidates, result = run_cached(
            params["n_customers"], params["n_candidates"], params["max_distance"],
            params["orders_per_customer_day"], params["contribution_per_order"],
            params["delivery_cost_per_km"], params["store_cost_per_day"],
            params["max_store_cap"], params["max_orders_per_store_day"],
        )
        st.session_state["dark_store_result"] = {
            "customers": customers,
            "candidates": candidates,
            "result": result,
            "params": params,
            "completed_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

saved = st.session_state["dark_store_result"]
customers = saved["customers"]
candidates = saved["candidates"]
result = saved["result"]
used = saved["params"]

if used != params:
    st.info(
        "The sidebar inputs differ from the values used for the displayed KPI table and map. "
        "Click **Run optimization & refresh KPIs/map** to apply the new inputs. "
        "The result table intentionally shows the last completed run until you submit the form."
    )

st.caption(
    f"KPI table last refreshed: {saved.get('completed_at', 'session start')} · "
    f"{used['n_customers']:,} customers · {used['n_candidates']} candidate sites · "
    f"{used['max_distance']:.1f} km service limit · {used['max_orders_per_store_day']:,.0f} orders/store/day capacity · at most {used['max_store_cap']} stores. "
    "Customer and candidate coordinates are synthetic and reproducible."
)

kpis = result["kpis"]
left, right = st.columns([1.0, 1.3], gap="large")
with left:
    st.markdown("<div class='section-label'>1. Key performance indicators</div>", unsafe_allow_html=True)
    kpi_df = render_kpi_table(kpis, used["max_distance"], used["max_orders_per_store_day"])
    st.dataframe(kpi_df, use_container_width=True, hide_index=True, height=470)
    st.download_button(
        "Download all customer assignments (CSV)",
        data=result["assignments"].to_csv(index=False).encode("utf-8"),
        file_name="chennai_dark_store_customer_assignments.csv",
        mime="text/csv",
        use_container_width=True,
    )
with right:
    st.markdown("<div class='section-label'>2. Required profit sweep: 2, 5, 10, 20 and 25 stores</div>", unsafe_allow_html=True)
    # Reuse the current run's scenario curve when the main settings are exactly
    # the assignment benchmark; otherwise run a dedicated, fixed 10k x 50 sweep.
    if used["n_customers"] == 10000 and used["n_candidates"] == 50:
        scenario = result["scenario_df"].copy()
    else:
        with st.spinner("Running the required 10,000-customer / 50-candidate sweep…"):
            scenario = run_assignment_sweep_cached(
                used["max_distance"],
                used["orders_per_customer_day"],
                used["contribution_per_order"],
                used["delivery_cost_per_km"],
                used["store_cost_per_day"],
                used["max_orders_per_store_day"],
            ).copy()
    scenario["Profit (₹ lakh/day)"] = scenario["Daily profit (₹)"] / 100000.0
    fig = px.line(
        scenario,
        x="Store cap",
        y="Profit (₹ lakh/day)",
        markers=True,
        text="Stores opened",
        hover_data={
            "Effective cap": True,
            "Stores opened": True,
            "Customers served": ":,",
            "Daily profit (₹)": ":,.0f",
            "Profit (₹ lakh/day)": False,
        },
        labels={"Store cap": "Maximum stores allowed", "Profit (₹ lakh/day)": "Net expected daily profit (₹ lakh)"},
        title=(
            f"Fixed benchmark: 10,000 customers · 50 candidates · {used['max_distance']:.1f} km limit"
            f" · {used['max_orders_per_store_day']:,.0f} orders/store/day capacity"
        ),
    )
    fig.update_traces(line_width=3, marker_size=10, textposition="top center")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=65, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    sweep_table = scenario[["Store cap", "Effective cap", "Stores opened", "Customers served", "Daily profit (₹)"]].rename(
        columns={"Daily profit (₹)": "Net profit / day (₹)"}
    )
    st.dataframe(
        sweep_table.style.format({"Net profit / day (₹)": "₹{:,.0f}", "Customers served": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download five-case profit sweep (CSV)",
        data=sweep_table.to_csv(index=False).encode("utf-8"),
        file_name="dark_store_profit_sweep_10000_customers_50_candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "This sweep always compares the five required caps on the same 10,000 synthetic customers and 50 candidate sites. "
        "The distance limit, per-store order capacity and economics come from the last completed sidebar run. A cap means ‘up to’ that many stores."
    )

st.markdown("---")
st.markdown("<div class='section-label'>3. Detailed geographic solution</div>", unsafe_allow_html=True)
st.caption(
    "The map shows every candidate location, open stores, service-radius circles, a sample of customer points, "
    "and a sample of assignment links. All customers are included in the optimization and CSV; customer markers/lines are sampled for map responsiveness. "
    "Distances are straight-line Haversine kilometres rather than road-network distances."
)
map_obj = make_solution_map(customers, candidates, result, used["max_distance"])
st_folium(map_obj, use_container_width=True, height=650, key="chennai_solution_map")

st.markdown("#### Open-store allocation details")
if result["store_metrics"].empty:
    st.warning("No dark stores were opened under the current economics; the best feasible decision is to remain closed.")
else:
    store_table = result["store_metrics"].copy()
    st.dataframe(
        store_table.style.format({
            "Latitude": "{:.5f}", "Longitude": "{:.5f}",
            "Expected orders/day": "{:,.1f}",
            "Order capacity/day": "{:,.0f}",
            "Capacity utilization (%)": "{:.1f}%",
            "Customer contribution/day (₹)": "₹{:,.0f}",
            "Operating cost/day (₹)": "₹{:,.0f}",
            "Store profit contribution/day (₹)": "₹{:,.0f}",
            "Mean delivery distance (km)": "{:.2f}",
            "Max delivery distance (km)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download open-store summary (CSV)",
        data=store_table.to_csv(index=False).encode("utf-8"),
        file_name="chennai_dark_store_open_store_summary.csv",
        mime="text/csv",
    )

with st.expander("Model formulation, constraint and assumptions"):
    st.markdown("**Decision variables**")
    st.latex(r"y_j\in\{0,1\}:\text{ 1 if candidate site }j\text{ is opened};\quad x_{ij}\in\{0,1\}:\text{ 1 if customer }i\text{ is served by site }j.")
    st.markdown("**Objective: maximize expected daily net profit**")
    st.latex(r"\max\quad \sum_i\sum_j x_{ij}\,q_i\bigl(m-c_d d_{ij}\bigr)-f\sum_j y_j")
    st.markdown("**Constraints, including the requested maximum-distance restriction**")
    st.latex(r"\sum_j x_{ij}\leq 1\quad\forall i;\qquad x_{ij}\leq y_j\quad\forall i,j;")
    st.latex(r"x_{ij}=0\quad\text{if }d_{ij}>D_{\max};\qquad\sum_j y_j\leq K;")
    st.latex(r"\sum_i q_i x_{ij}\leq C y_j\quad\forall j.")
    st.markdown(
        "Here, `q` is expected orders per customer per day, `m` is contribution per order before delivery, "
        "`c_d` is delivery cost per order per km, `d` is Haversine distance, `f` is daily store operating cost, "
        "`D_max` is the permitted customer–store distance, `K` is the store-count cap, and `C` is the maximum orders each store can process per day. "
        "The capacity constraint prevents assignments to an open store from exceeding its order throughput. Because each synthetic customer has the same expected order rate, the app enforces this as an integer customer limit per store, floor(C/q). "
        "Customers outside every open store's radius, or who cannot be accommodated because stores are at capacity, remain unserved. "
        "For a chosen set of stores, assignment uses a deterministic capacity-feasible greedy allocation; location selection uses greedy profit-improving additions/removals under that allocation. This is a heuristic, not a proof of global optimality."
    )
    st.warning(
        "The facility-selection problem is combinatorial. This app uses a capacity-aware greedy selection heuristic with profitable removals; "
        "it produces a capacity-feasible solution but does not guarantee the global optimum. Scenario runs warm-start from the previous cap, so reported profit is non-decreasing as the cap increases."
    )
    st.markdown("**Default assumptions**: 0.18 orders/customer/day; 300 orders/store/day capacity; ₹120 contribution/order before delivery; ₹5 per order per delivery-km; ₹3,500 daily fixed cost per open dark store. "
                "The data are synthetic for Chennai and should be replaced with geocoded customers, site costs and demand data for real decisions.")

st.markdown("---")
st.markdown(
    "<p class='small-note'>Assignment prototype · Built with Streamlit, NumPy, Pandas, Plotly and Folium. "
    "This is a planning model for educational use; profit values are estimates under the assumptions above.</p>",
    unsafe_allow_html=True,
)
