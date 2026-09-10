"""
ARJUN — Unified End-to-End Cyber Defence World Model Orchestrator

This script executes the complete lifecycle of proactive cyber defence:
1. Verifies environment & initializes database
2. Ensures sample telemetry exists or ingests user-provided data
3. Runs the multi-modal telemetry pipeline (CSV, PCAP, Zeek conn.log)
4. Constructs temporal state representations and dynamic communication graphs
5. Loads the trained Hybrid World Model (GNN + LSTM)
6. Performs K-step autoregressive future network state simulation
7. Maps projected states to MITRE ATT&CK stages & calculates proactive risk
8. Dispatches security alerts and logs full audit trail into SQLite database
9. Displays an executive SOC Intelligence Summary
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.connection import init_database, DATABASE_PATH
from database.repository import (
    create_analysis_record,
    update_analysis_record,
    create_forecast_record,
    update_forecast_record,
    save_risk_record,
)
from pipeline import run_pipeline
from world_model.hybrid_wrapper import HybridWorldModelWrapper
from forecasting.hybrid_k_step_forecaster import HybridKStepForecaster
from risk.unified_analysis import UnifiedAnalysisService
from generate_sample_data import (
    generate_sample_csv,
    generate_sample_pcap,
    generate_sample_zeek,
)


def calculate_attack_probability(state, feature_names):
    """
    Computes bounded attack probability [0, 1] based on key cyber indicators.
    """
    state = np.asarray(state, dtype=np.float32)
    values = dict(zip(feature_names, state))

    def get(name):
        return float(values.get(name, 0.0))

    port_scan = get("port_scan_count")
    retransmissions = get("retransmission_count")
    rst_count = get("rst_count")
    unique_ports = max(get("unique_dst_ports"), get("avg_unique_dst_ports"))
    unique_hosts = max(get("unique_dst_ips"), get("avg_unique_dst_hosts"))
    syn_count = get("syn_count")
    bytes_per_sec = get("avg_bytes_per_second")

    scan_score = min(port_scan / 10.0, 1.0)
    port_score = min(unique_ports / 50.0, 1.0)
    host_score = min(unique_hosts / 20.0, 1.0)
    syn_score = min(syn_count / 100.0, 1.0)
    retx_score = min(retransmissions / 50.0, 1.0)
    rst_score = min(rst_count / 50.0, 1.0)
    traffic_score = min(bytes_per_sec / 1_000_000.0, 1.0)

    prob = (
        0.25 * scan_score
        + 0.15 * port_score
        + 0.15 * host_score
        + 0.15 * syn_score
        + 0.10 * retx_score
        + 0.10 * rst_score
        + 0.10 * traffic_score
    )
    return float(np.clip(prob, 0.0, 1.0))


def run_end_to_end(
    input_path: str,
    model_path: str = "saved_models/hybrid_world_model.pt",
    steps: int = 10,
    window_seconds: int = 5,
    sequence_length: int = 5,
    save_db: bool = True,
):
    print("\n" + "=" * 75)
    print("      ARJUN PROACTIVE CYBER DEFENCE WORLD MODEL -- END-TO-END RUNNER")
    print("=" * 75)
    print(f"[*] Timestamp       : {datetime.now(timezone.utc).isoformat()}")
    print(f"[*] Input Path      : {input_path}")
    print(f"[*] Model Path      : {model_path}")
    print(f"[*] Forecast Steps  : {steps} lookahead windows")
    print(f"[*] Time Window     : {window_seconds}s per state")
    print(f"[*] Sequence Length : {sequence_length} historical states")
    print("-" * 75)

    # 1. Database Initialization
    print("[1/6] Initializing ARJUN Database...")
    init_database()
    print(f"      SQLite Database online at: {DATABASE_PATH}")

    # Verify input exists
    input_file = Path(input_path)
    if not input_file.exists():
        sample_dir = ROOT / "data" / "sample"
        print(f"[!] Input file {input_path} not found. Generating default sample telemetry...")
        generate_sample_csv(sample_dir / "sample_network_flows.csv")
        generate_sample_pcap(sample_dir / "sample_traffic.pcap")
        generate_sample_zeek(sample_dir / "sample_zeek_conn.log")
        input_file = sample_dir / "sample_network_flows.csv"
        print(f"      Proceeding with: {input_file}")

    analysis_id = None
    forecast_id = None

    if save_db:
        suffix = input_file.suffix.lower().lstrip(".")
        analysis_id = create_analysis_record(filename=input_file.name, input_type=suffix)
        forecast_id = create_forecast_record(
            filename=input_file.name, steps=steps, sequence_length=sequence_length
        )

    # 2. Pipeline Execution
    print("\n[2/6] Running Telemetry Pipeline (Ingestion -> Features -> Graphs -> States)...")
    try:
        pipeline_result = run_pipeline(
            str(input_file),
            window_seconds=window_seconds,
            sequence_length=sequence_length,
        )
    except Exception as exc:
        if save_db and analysis_id:
            update_analysis_record(analysis_id, status="failed", error_message=str(exc))
            update_forecast_record(forecast_id, status="failed", error_message=str(exc))
        raise

    states = np.asarray(pipeline_result["states"], dtype=np.float32)
    graph_sequences = pipeline_result["graph_sequences"]
    feature_names = list(pipeline_result["feature_names"])
    effective_seq_len = int(pipeline_result["sequence_length"])
    features_df = pipeline_result.get("features", pipeline_result.get("data"))

    print(f"      Extracted {len(features_df)} network flow events")
    print(f"      Generated {len(states)} temporal network states (Dimension: {states.shape[1]})")
    print(f"      Constructed {len(graph_sequences)} aligned communication graph sequences")

    if save_db and analysis_id:
        update_analysis_record(
            analysis_id,
            state_count=len(states),
            state_dimension=states.shape[1],
            feature_count=len(feature_names),
            status="completed",
        )

    if len(states) < effective_seq_len:
        raise ValueError(f"Need at least {effective_seq_len} states for sequence modeling; got {len(states)}")

    # 3. Neural World Model Loading
    print("\n[3/6] Loading Pre-trained Hybrid Neural World Model (GNN + LSTM)...")
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    world_model = HybridWorldModelWrapper.from_checkpoint(
        model_file, state_dimension=states.shape[1], device="cpu"
    )
    print(f"      Model weights loaded successfully from: {model_file.name}")

    # 4. K-Step Autoregressive Simulation
    print(f"\n[4/6] Simulating Future Cyber States ({steps} steps lookahead)...")
    current_state = states[-1]
    current_sequence = states[-effective_seq_len:]
    current_graph_sequence = graph_sequences[-1]

    forecaster = HybridKStepForecaster(world_model, device="cpu")
    future_states = forecaster.forecast(
        current_sequence, current_graph_sequence, steps=steps
    )
    future_states = np.asarray(future_states, dtype=np.float32)
    print(f"      Successfully predicted {len(future_states)} future states into the horizon")

    # 5. MITRE ATT&CK & Proactive Risk Intelligence
    print("\n[5/6] Evaluating MITRE ATT&CK Stage Progression & Proactive Risk Scores...")
    current_prob = calculate_attack_probability(current_state, feature_names)
    future_probs = [calculate_attack_probability(fs, feature_names) for fs in future_states]

    intelligence = UnifiedAnalysisService(feature_names)
    analysis = intelligence.analyze(
        current_state=current_state,
        future_states=future_states,
        current_attack_probability=current_prob,
        future_attack_probabilities=np.asarray(future_probs, dtype=np.float32),
    )

    # 6. Database Persistence
    if save_db and forecast_id:
        update_forecast_record(forecast_id, status="completed")
        # Save risk steps
        timeline = analysis["timeline"]
        for item in timeline:
            atk_prob = float(
                item.get("future_attack_probability")
                if item.get("future_attack_probability") is not None
                else item.get("attack_probability", item.get("current_attack_probability", 0.0))
            )
            save_risk_record(
                analysis_id=analysis_id,
                step=item["step"],
                risk_score=item["risk_score"],
                risk_level=item["risk_level"],
                attack_probability=atk_prob,
                mitre_stage=item["mitre_stage"],
                mitre_confidence=item.get("mitre_confidence", 0.85),
            )
        print(f"      Logged {len(timeline)} risk trajectory steps into SQLite database")

    # 7. Executive Summary Presentation
    summary = analysis["summary"]
    peak_step = summary.get("highest_risk_step", 1)
    peak_risk_raw = summary.get("maximum_forecast_risk", 0.0)
    peak_risk = peak_risk_raw * 100.0 if peak_risk_raw <= 1.0 else peak_risk_raw
    overall_level = summary.get("maximum_risk_level", "LOW")

    contributions = analysis.get("feature_contributions", [])
    top_drivers = []
    if contributions and isinstance(contributions, list):
        first_item = contributions[0]
        if isinstance(first_item, dict) and "features" in first_item:
            feat_list = first_item["features"]
            top_drivers = [f["feature"] for f in feat_list[:3] if isinstance(f, dict) and "feature" in f]
        elif isinstance(first_item, dict) and "feature" in first_item:
            top_drivers = [c["feature"] for c in contributions[:3] if isinstance(c, dict) and "feature" in c]
    top_driver_str = ", ".join(top_drivers) if top_drivers else "Traffic Volume / Flow Anomalies"

    print("\n" + "=" * 75)
    print("                    EXECUTIVE SOC INTELLIGENCE REPORT")
    print("=" * 75)
    print(f" Overall Risk Level    : {overall_level}")
    print(f" Peak Risk Score       : {peak_risk:.1f}/100 at Step T+{peak_step}")
    print(f" Primary Threat Driver : {top_driver_str}")
    print(f" Mitre ATT&CK Stage    : {summary.get('highest_risk_mitre_stage', 'Unknown')}")
    print("-" * 75)
    print(f" {'Step':<6} | {'Lookahead':<12} | {'Atk Prob':<10} | {'Risk':<8} | {'Level':<10} | {'MITRE ATT&CK Stage':<22}")
    print("-" * 75)

    for item in analysis["timeline"]:
        step = item["step"]
        label = "T+0 (Now)" if step == 0 else f"T+{step} (+{step*window_seconds}s)"
        atk_prob = float(
            item.get("future_attack_probability")
            if item.get("future_attack_probability") is not None
            else item.get("attack_probability", item.get("current_attack_probability", 0.0))
        )
        prob_str = f"{atk_prob*100:.1f}%"
        risk_str = f"{item['risk_score']*100:.1f}" if item['risk_score'] <= 1.0 else f"{item['risk_score']:.1f}"
        level = item["risk_level"]
        stage = item["mitre_stage"]
        print(f" {step:<6} | {label:<12} | {prob_str:<10} | {risk_str:<8} | {level:<10} | {stage:<22}")

    print("=" * 75)
    print(" [SUCCESS] ARJUN End-to-End Proactive Defence Execution Completed Successfully.")
    print("=" * 75 + "\n")

    return {
        "analysis_id": analysis_id,
        "forecast_id": forecast_id,
        "summary": summary,
        "timeline": analysis["timeline"],
        "mitre_results": analysis["mitre_results"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="ARJUN Proactive Cyber Defence World Model Orchestrator"
    )
    parser.add_argument(
        "--input",
        default="data/sample/sample_network_flows.csv",
        help="Input telemetry file (CSV, PCAP, or Zeek conn.log)",
    )
    parser.add_argument(
        "--model",
        default="saved_models/hybrid_world_model.pt",
        help="Hybrid World Model checkpoint path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of future state forecasting steps",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Aggregation window duration in seconds",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of historical states in temporal sequence",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database persistence",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate synthetic sample telemetry before running",
    )

    args = parser.parse_args()

    if args.generate_samples:
        sample_dir = ROOT / "data" / "sample"
        generate_sample_csv(sample_dir / "sample_network_flows.csv")
        generate_sample_pcap(sample_dir / "sample_traffic.pcap")
        generate_sample_zeek(sample_dir / "sample_zeek_conn.log")

    run_end_to_end(
        input_path=args.input,
        model_path=args.model,
        steps=args.steps,
        window_seconds=args.window,
        sequence_length=args.sequence_length,
        save_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
