"""
Streamlit demo UI for the podcast recommendation system.

Three tabs:
  1. Recommendations — interactive recommendation explorer
  2. Experiments    — compare MLflow runs side by side
  3. Monitoring     — drift and performance dashboard

Run:
  streamlit run src/ui/app.py
"""

import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Podcast RecSys",
    page_icon="🎙",
    layout="wide",
)

st.title("Podcast Recommendation System")
st.caption("Sentence-BERT embeddings · Collaborative filtering · LightGBM ranker · MLOps pipeline")

tab1, tab2, tab3 = st.tabs(["Recommendations", "Experiments", "Monitoring"])


# ── Tab 1: Recommendations ────────────────────────────────────────────────────

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Query")
        user_id = st.text_input("User ID", value="user_0001")
        query = st.text_area(
            "Interest description",
            value="technology and artificial intelligence",
            height=100,
            help="Free-text query. The embedder finds semantically matching podcasts.",
        )
        top_k = st.slider("Number of recommendations", 1, 20, 10)
        exclude = st.text_area(
            "Exclude podcast IDs (one per line)",
            value="",
            height=80,
        )
        submit = st.button("Get recommendations", type="primary", use_container_width=True)

    with col2:
        st.subheader("Results")
        if submit:
            exclude_ids = [e.strip() for e in exclude.splitlines() if e.strip()]
            payload = {
                "user_id": user_id,
                "query": query,
                "top_k": top_k,
                "exclude_podcast_ids": exclude_ids,
            }
            with st.spinner("Fetching recommendations..."):
                try:
                    resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()

                    st.caption(f"Latency: {data['latency_ms']:.1f} ms")

                    for i, rec in enumerate(data["recommendations"], 1):
                        with st.container():
                            score_pct = int(rec["ranker_score"] * 100)
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.markdown(f"**{i}. {rec['title']}**")
                                st.caption(f"{rec['category']} · {rec['description'][:150]}...")
                            with c2:
                                st.metric("Score", f"{score_pct}%")
                            st.divider()

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the API. "
                        "Make sure it's running: `uvicorn src.api.main:app --reload`"
                    )
                except Exception as e:
                    st.error(f"API error: {e}")
        else:
            st.info("Enter a user ID and query, then click 'Get recommendations'.")

    st.divider()
    st.subheader("Find similar podcasts")
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        podcast_id = st.text_input("Podcast ID", value="pod_0001")
        sim_k = st.slider("Number of similar", 1, 20, 5)
        sim_submit = st.button("Find similar", use_container_width=True)
    with sim_col2:
        if sim_submit:
            try:
                resp = requests.get(
                    f"{API_URL}/similar/{podcast_id}",
                    params={"top_k": sim_k},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                st.caption(f"Similar to: **{data['source_podcast']['title']}**")
                df = pd.DataFrame(data["similar"])
                st.dataframe(
                    df[["podcast_id", "title", "category", "embedding_score"]],
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error: {e}")


# ── Tab 2: Experiments ────────────────────────────────────────────────────────

with tab2:
    st.subheader("MLflow experiment runs")
    st.caption("Compare hyperparameters and metrics across training runs.")

    mlflow_uri = st.text_input("MLflow tracking URI", value="http://localhost:5000")
    load_runs = st.button("Load runs")

    if load_runs:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("podcast-recsys")

            if experiment is None:
                st.warning("No 'podcast-recsys' experiment found. Run training first.")
            else:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.ndcg_at_10 DESC"],
                    max_results=20,
                )
                if not runs:
                    st.info("No runs yet. Train the pipeline first.")
                else:
                    rows = []
                    for run in runs:
                        rows.append({
                            "run_id": run.info.run_id[:8],
                            "status": run.info.status,
                            "ndcg@10": run.data.metrics.get("ndcg_at_10", None),
                            "cf_rmse": run.data.metrics.get("cf_rmse", None),
                            "n_factors": run.data.params.get("n_factors", None),
                            "n_estimators": run.data.params.get("ranker_n_estimators", None),
                            "lr": run.data.params.get("ranker_learning_rate", None),
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)

                    if "ndcg@10" in df.columns and df["ndcg@10"].notna().any():
                        st.line_chart(df.set_index("run_id")["ndcg@10"])

        except ImportError:
            st.error("MLflow not installed. Run: pip install mlflow")
        except Exception as e:
            st.error(f"Could not connect to MLflow: {e}")

    st.divider()
    st.subheader("API health")
    if st.button("Check API status"):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            data = resp.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", data["status"].upper())
            c2.metric("Uptime", f"{data['uptime_seconds']:.0f}s")
            c3.metric("Requests served", data["requests_served"])
        except Exception as e:
            st.error(f"API unreachable: {e}")


# ── Tab 3: Monitoring ─────────────────────────────────────────────────────────

with tab3:
    st.subheader("Drift and performance monitoring")

    reports_dir = Path("reports/")
    drift_files = sorted(reports_dir.glob("**/drift_summary*.json"), reverse=True)
    perf_files = sorted(reports_dir.glob("**/performance_summary.json"), reverse=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Data drift")
        if drift_files:
            with open(drift_files[0]) as f:
                drift = json.load(f)
            status = "DRIFT DETECTED" if drift.get("has_drift") else "No drift"
            color = "red" if drift.get("has_drift") else "green"
            st.markdown(f"**Status:** :{color}[{status}]")
            st.json({
                "drift_share": drift.get("drift_share"),
                "threshold": drift.get("drift_threshold"),
                "reference_rows": drift.get("n_reference"),
                "current_rows": drift.get("n_current"),
                "timestamp": drift.get("timestamp"),
            })
        else:
            st.info("No drift reports found. Run `python -m src.monitoring.monitor` to generate one.")

    with col2:
        st.markdown("#### Model performance")
        if perf_files:
            with open(perf_files[0]) as f:
                perf = json.load(f)
            decay = perf.get("has_performance_decay", False)
            status = "DECAY DETECTED" if decay else "Stable"
            color = "red" if decay else "green"
            st.markdown(f"**Status:** :{color}[{status}]")
            st.json({
                "baseline_ndcg": perf.get("baseline_ndcg"),
                "current_ndcg": perf.get("current_ndcg"),
                "drop_pct": perf.get("relative_drop_pct"),
                "threshold_pct": perf.get("performance_threshold_pct"),
            })
        else:
            st.info("No performance reports found yet.")

    st.divider()
    st.subheader("Simulate drift check")
    st.caption("Run a quick drift check on synthetic data to see how the monitor works.")
    if st.button("Run drift simulation"):
        import numpy as np
        from src.monitoring.monitor import RecommendationMonitor

        rng = np.random.default_rng(42)
        reference = pd.DataFrame({
            "embedding_score": rng.normal(0.5, 0.15, 1000),
            "collab_score": rng.normal(0.6, 0.12, 1000),
            "play_count_norm": rng.beta(2, 5, 1000),
        })
        current = pd.DataFrame({
            "embedding_score": rng.normal(0.62, 0.20, 300),
            "collab_score": rng.normal(0.50, 0.16, 300),
            "play_count_norm": rng.beta(2, 3, 300),
        })

        with st.spinner("Running drift analysis..."):
            monitor = RecommendationMonitor(drift_threshold=0.15)
            summary = monitor.run_drift_report(reference, current, output_dir="reports/sim/")

        if summary.get("has_drift"):
            st.error(f"Drift detected! Score: {summary.get('drift_share', summary.get('avg_psi', 'N/A'))}")
        else:
            st.success("No significant drift detected.")
        st.json(summary)
