"""
ARJUN — Unified Services Launcher

Starts both the FastAPI REST Backend and Streamlit Enterprise SOC Dashboard
with a single command, allowing complete end-to-end operation.

Usage:
    python start_services.py
    python start_services.py --backend-only
    python start_services.py --frontend-only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def start_backend(host="127.0.0.1", port=8000):
    print(f"[*] Launching FastAPI Backend on http://{host}:{port}...")
    cmd = [
        PYTHON_EXE,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT))


def start_frontend(port=8501):
    dashboard_path = ROOT / "frontend" / "streamlit_dashboard.py"
    print(f"[*] Launching Streamlit SOC Dashboard on http://localhost:{port}...")
    cmd = [
        PYTHON_EXE,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="ARJUN Cyber Defence Services Launcher"
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Run only the FastAPI backend service",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Run only the Streamlit SOC dashboard",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Port for FastAPI (default: 8000)",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=8501,
        help="Port for Streamlit (default: 8501)",
    )

    args = parser.parse_args()

    processes = []

    print("\n" + "=" * 65)
    print("      ARJUN PROACTIVE CYBER DEFENCE -- SERVICES LAUNCHER")
    print("=" * 65)

    try:
        if not args.frontend_only:
            p_backend = start_backend(port=args.backend_port)
            processes.append(("FastAPI Backend", p_backend))

        if not args.backend_only:
            time.sleep(1)
            p_frontend = start_frontend(port=args.frontend_port)
            processes.append(("Streamlit SOC", p_frontend))

        print("-" * 65)
        print(" [OK] Services are active:")
        if not args.frontend_only:
            print(f"      - API Swagger UI : http://127.0.0.1:{args.backend_port}/docs")
            print(f"      - API Health     : http://127.0.0.1:{args.backend_port}/api/health")
        if not args.backend_only:
            print(f"      - SOC Dashboard  : http://localhost:{args.frontend_port}")
        print("=" * 65)
        print(" Press Ctrl+C to shut down all services gracefully.\n")

        while True:
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    print(f"[!] Process {name} exited with code {ret}")
                    return
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] Shutting down services...")
        for name, proc in processes:
            print(f"    Stopping {name}...")
            proc.terminate()
            proc.wait(timeout=5)
        print("[*] All services stopped cleanly.")


if __name__ == "__main__":
    main()
