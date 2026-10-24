# Chennai Dark Store Location Optimizer

A Streamlit application for an illustrative dark-store facility-location problem in Chennai.

## Features

- Inputs for customer count, candidate site count, maximum customer-to-store distance, store-count cap and economic assumptions.
- A distance-filtered customer assignment model that leaves customers unserved if no open store is within the permitted radius.
- KPI table, daily profit-versus-store-cap chart (caps 2, 5, 10, 20 and 25), interactive Folium map, store-level metrics, and customer-assignment CSV export.
- Synthetic, reproducible customer and candidate-site coordinates around Chennai.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new **public GitHub repository** and upload `app.py`, `optimizer.py`, `requirements.txt`, `.streamlit/config.toml`, and `README.md`.
2. Sign in at <https://share.streamlit.io> with the GitHub account that owns the repository.
3. Select **Create app**, choose the repository and branch, set the entrypoint to `app.py`, and deploy.
4. Once the app starts successfully, copy the public app URL ending in `.streamlit.app` and submit that URL.

Deployment requires access to the user's GitHub account and Streamlit Community Cloud workspace; this project bundle does not contain credentials.

## Model

Let `y_j` indicate whether candidate `j` opens and `x_ij` indicate whether customer `i` is assigned to candidate `j`. The objective maximizes expected daily contribution from served customers minus fixed daily costs for open stores:

`max sum(i,j) x_ij * q_i * (m - c_d * d_ij) - f * sum(j) y_j`

Subject to:

- Each customer is assigned to at most one store: `sum(j) x_ij <= 1`.
- Assignments require an open site: `x_ij <= y_j`.
- Maximum distance: `x_ij = 0` where `d_ij > D_max`.
- Store cap: `sum(j) y_j <= K`.
- `x_ij, y_j` are binary.

Distance is a straight-line Haversine distance, not a road-network travel distance. The facility-selection step uses greedy marginal-profit selection and add/remove/one-for-one-swap local search; it is feasible and locally improved, but does not guarantee a global optimum. For any selected facility set, the assignment is chosen to maximize customer contribution under the shared assumptions.

## Default economics

- 0.18 expected orders per customer per day.
- ₹120 contribution per order before delivery.
- ₹5 delivery cost per order per kilometre.
- ₹3,500 fixed operating cost per store per day.

These are editable illustrative assumptions, not market estimates. All coordinates and demand data are synthetic and reproducible.

## Reproducible baseline check

Using the default assumptions, 10,000 generated customers, 50 candidate sites, a 5 km service limit and a daily store cost of ₹3,500, the current heuristic benchmark returns:

| Maximum stores allowed | Stores opened | Customers served | Expected net profit/day |
|---:|---:|---:|---:|
| 2 | 2 | 5,743 | ₹102,245 |
| 5 | 5 | 8,332 | ₹143,307 |
| 10 | 10 | 9,851 | ₹158,422 |
| 20 | 10 | 9,851 | ₹158,422 |
| 25 | 10 | 9,851 | ₹158,422 |

The plateau after 10 stores means the extra allowed sites do not improve net profit under these assumed costs and generated demand—not that the model was forced to open every permitted store. Results may vary if assumptions or random-seed logic are changed.
