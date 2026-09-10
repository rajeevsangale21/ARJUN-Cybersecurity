"""
ARJUN - END-TO-END CONTROL CENTER

Single entry point for the working ARJUN prototype.

Modes
-----
check
    Validate the important project modules, model/artifact files, and the
    Zeek predictive path without changing production artifacts.

live
    Start the Zeek live collector. The collector performs:
        Zeek conn.log
          -> preprocessing/features
          -> 33-D state + graph
          -> trained Hybrid World Model
          -> K-step forecast
          -> risk + MITRE
          -> alert

dashboard
    Start the Streamlit SOC dashboard.

full
    Start the Zeek collector and Streamlit dashboard together.

historical
    Run the existing unified CSV/PCAP/Zeek inference entry point.

Examples
--------
    python run.py check
    python run.py live --input data/raw/logs/conn.log
    python run.py dashboard
    python run.py full --input data/raw/logs/conn.log
    python run.py historical --input data/raw/csv/example.csv
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PYTHON = sys.executable

PRODUCTION_WORLD_MODEL = ROOT / "saved_models" / "hybrid_world_model.pt"
XGBOOST_MODEL = ROOT / "saved_models" / "xgboost_attack_classifier.pkl"

REQUIRED_ARTIFACTS = [
    ROOT / "data" / "processed" / "training_states.npz",
    ROOT / "data" / "processed" / "graph_sequences_states.npz",
    ROOT / "data" / "processed" / "graph_sequences.pkl",
    PRODUCTION_WORLD_MODEL,
    XGBOOST_MODEL,
]

ZEek_COLLECTOR_CANDIDATES = [
    ROOT / "ingestion" / "zeek_live_collector_v7.py",
    ROOT / "ingestion" / "zeek_live_collector.py",
]

DASHBOARD = ROOT / "frontend" / "streamlit_dashboard.py"

CORE_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "joblib",
    "xgboost",
    "streamlit",
]


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run_process(
    command: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    print()
    print(">>", " ".join(str(x) for x in command))
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        capture_output=capture,
        text=True,
    )


def find_zeek_collector() -> Path:
    for candidate in ZEek_COLLECTOR_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No Zeek live collector found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in ZEek_COLLECTOR_CANDIDATES)
    )


def check_modules() -> list[str]:
    missing = []
    for name in CORE_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    return missing


def check_project_imports() -> list[str]:
    modules = [
        "ingestion.zeek_loader",
        "ingestion.zeek_to_state",
        "preprocessing.parser",
        "preprocessing.cleaner",
        "features.flow_features",
        "features.packet_features",
        "features.temporal_features",
        "features.behavioral_features",
        "features.contextual_features",
        "features.graph_builder",
        "world_model.state_encoder",
        "world_model.hybrid_world_model",
        "forecasting.hybrid_k_step_forecaster",
        "risk.risk_engine",
        "alerts.alert_engine",
    ]

    failures = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def check_artifacts() -> list[Path]:
    missing = []
    for path in REQUIRED_ARTIFACTS:
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
    return missing


def check_world_model_checkpoint() -> tuple[bool, str]:
    try:
        import torch

        from world_model.hybrid_world_model import HybridWorldModel

        checkpoint = torch.load(
            PRODUCTION_WORLD_MODEL,
            map_location="cpu",
            weights_only=False,
        )

        required = {"model_state_dict", "state_dimension"}
        missing = required.difference(checkpoint.keys())
        if missing:
            return False, f"checkpoint missing keys: {sorted(missing)}"

        state_dimension = int(checkpoint["state_dimension"])
        if state_dimension != 33:
            return False, f"expected state dimension 33, got {state_dimension}"

        model = HybridWorldModel(
            state_dimension=33,
            graph_input_dimension=6,
            hidden_dimension=64,
            lstm_layers=1,
            residual_scale=0.75,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return True, (
            f"checkpoint OK | state_dim={state_dimension} | "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_state_dataset() -> tuple[bool, str]:
    try:
        import numpy as np

        path = ROOT / "data" / "processed" / "training_states.npz"
        with np.load(path, allow_pickle=True) as data:
            if "states" not in data:
                return False, "training_states.npz missing 'states'"
            states = np.asarray(data["states"])
            if states.ndim != 2 or states.shape[1] != 33:
                return False, f"invalid state shape: {states.shape}"
            if not np.isfinite(states).all():
                return False, "state dataset contains non-finite values"

            labels_key = "labels" if "labels" in data else "y" if "y" in data else None
            label_count = len(data[labels_key]) if labels_key else 0

        return True, (
            f"states={states.shape} | labels={label_count:,} | finite=YES"
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_zeek_path() -> tuple[bool, str]:
    try:
        collector = find_zeek_collector()
        from ingestion.zeek_loader import load_zeek_conn

        return True, f"collector={collector.relative_to(ROOT)} | Zeek loader import OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def command_check(args) -> int:
    header("ARJUN END-TO-END SYSTEM CHECK")

    print("\n[1/5] Python dependencies")
    missing = check_modules()
    if missing:
        for item in missing:
            print(f"FAIL  {item}")
    else:
        print("PASS  All core dependencies import successfully.")

    print("\n[2/5] Project modules")
    failures = check_project_imports()
    if failures:
        for item in failures:
            print(f"FAIL  {item}")
    else:
        print("PASS  Core ARJUN modules import successfully.")

    print("\n[3/5] Production artifacts")
    missing_artifacts = check_artifacts()
    if missing_artifacts:
        for path in missing_artifacts:
            print(f"FAIL  {path.relative_to(ROOT)}")
    else:
        for path in REQUIRED_ARTIFACTS:
            print(f"PASS  {path.relative_to(ROOT)}")

    print("\n[4/5] World Model")
    wm_ok, wm_message = check_world_model_checkpoint()
    print(("PASS  " if wm_ok else "FAIL  ") + wm_message)

    print("\n[5/5] Network state + Zeek path")
    state_ok, state_message = check_state_dataset()
    print(("PASS  " if state_ok else "FAIL  ") + state_message)

    zeek_ok, zeek_message = check_zeek_path()
    print(("PASS  " if zeek_ok else "FAIL  ") + zeek_message)

    all_ok = (
        not missing
        and not failures
        and not missing_artifacts
        and wm_ok
        and state_ok
        and zeek_ok
    )

    print()
    print("=" * 78)
    print("END-TO-END CHECK:", "PASSED" if all_ok else "FAILED")
    print("=" * 78)

    if not all_ok:
        print(
            "\nFix the failed item(s) above before using the full live command."
        )
        return 1

    print(
        "\nWorking architecture:"
        "\n  Zeek/PCAP -> state S_t + graph G_t -> World Model"
        "\n  -> K-step forecast -> risk/MITRE -> alert -> dashboard"
    )
    return 0


def command_live(args) -> int:
    header("ARJUN LIVE ZEEK MODE")

    collector = find_zeek_collector()

    if not args.input:
        raise SystemExit(
            "Live mode requires --input pointing to a Zeek conn.log file."
        )

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        raise SystemExit(f"Zeek input does not exist: {input_path}")

    command = [
        PYTHON,
        str(collector),
        "--input",
        str(input_path),
    ]

    if args.interval is not None:
        command += ["--interval", str(args.interval)]

    if args.forecast_steps is not None:
        command += ["--forecast-steps", str(args.forecast_steps)]

    if args.no_email:
        command += ["--no-email"]

    return run_process(command).returncode


def command_dashboard(args) -> int:
    header("ARJUN STREAMLIT SOC DASHBOARD")

    if not DASHBOARD.exists():
        raise SystemExit(f"Dashboard not found: {DASHBOARD}")

    command = [
        PYTHON,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
    ]

    return run_process(command).returncode


def command_historical(args) -> int:
    header("ARJUN HISTORICAL INFERENCE")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    script = ROOT / "unified_forecast.py"
    if not script.exists():
        raise SystemExit(f"Historical inference entry point not found: {script}")

    command = [
        PYTHON,
        str(script),
        "--input",
        str(input_path),
        "--steps",
        str(args.forecast_steps),
        "--sequence-length",
        str(args.sequence_length),
    ]

    return run_process(command).returncode


def command_full(args) -> int:
    header("ARJUN FULL SOC MODE")

    collector = find_zeek_collector()

    if not args.input:
        raise SystemExit(
            "Full mode requires --input pointing to a Zeek conn.log file."
        )

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        raise SystemExit(f"Zeek input does not exist: {input_path}")

    collector_command = [
        PYTHON,
        str(collector),
        "--input",
        str(input_path),
    ]

    if args.interval is not None:
        collector_command += ["--interval", str(args.interval)]

    if args.forecast_steps is not None:
        collector_command += ["--forecast-steps", str(args.forecast_steps)]

    if args.no_email:
        collector_command += ["--no-email"]

    dashboard_command = [
        PYTHON,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
    ]

    print("\nStarting Zeek -> ARJUN collector...")
    collector_process = subprocess.Popen(
        collector_command,
        cwd=ROOT,
    )

    print("Starting Streamlit SOC dashboard...")
    time.sleep(2.0)

    dashboard_process = subprocess.Popen(
        dashboard_command,
        cwd=ROOT,
    )

    print()
    print("=" * 78)
    print("ARJUN FULL MODE RUNNING")
    print("=" * 78)
    print(f"Collector PID : {collector_process.pid}")
    print(f"Dashboard PID : {dashboard_process.pid}")
    print()
    print("Pipeline:")
    print("  Zeek conn.log")
    print("      -> 5-second state windows")
    print("      -> 33-D state + graph")
    print("      -> Hybrid World Model")
    print("      -> 5-step forecast")
    print("      -> risk + MITRE")
    print("      -> alert engine")
    print("      -> Streamlit SOC")
    print()
    print("Press Ctrl+C to stop both processes.")

    try:
        while True:
            collector_code = collector_process.poll()
            dashboard_code = dashboard_process.poll()

            if collector_code is not None:
                print(f"\nCollector exited with code {collector_code}.")
                break

            if dashboard_code is not None:
                print(f"\nDashboard exited with code {dashboard_code}.")
                break

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping ARJUN full mode...")

    finally:
        for process, name in (
            (collector_process, "collector"),
            (dashboard_process, "dashboard"),
        ):
            if process.poll() is None:
                print(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="ARJUN end-to-end control center."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "check",
        help="Validate dependencies, imports, artifacts and World Model.",
    )

    live = sub.add_parser(
        "live",
        help="Run the live Zeek -> ARJUN predictive pipeline.",
    )
    live.add_argument("--input", required=True)
    live.add_argument("--interval", type=float, default=None)
    live.add_argument("--forecast-steps", type=int, default=5)
    live.add_argument("--no-email", action="store_true")

    dash = sub.add_parser(
        "dashboard",
        help="Launch the Streamlit SOC dashboard.",
    )
    dash.add_argument("--unused", default=None, help=argparse.SUPPRESS)

    historical = sub.add_parser(
        "historical",
        help="Run CSV/PCAP/Zeek historical inference.",
    )
    historical.add_argument("--input", required=True)
    historical.add_argument("--forecast-steps", type=int, default=5)
    historical.add_argument("--sequence-length", type=int, default=10)

    full = sub.add_parser(
        "full",
        help="Run the Zeek collector and Streamlit dashboard together.",
    )
    full.add_argument("--input", required=True)
    full.add_argument("--interval", type=float, default=2.0)
    full.add_argument("--forecast-steps", type=int, default=5)
    full.add_argument("--no-email", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        return command_check(args)
    if args.command == "live":
        return command_live(args)
    if args.command == "dashboard":
        return command_dashboard(args)
    if args.command == "historical":
        return command_historical(args)
    if args.command == "full":
        return command_full(args)

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
