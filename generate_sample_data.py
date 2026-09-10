"""
ARJUN - Synthetic Sample Telemetry Generator

Generates realistic network telemetry in all three supported formats:
1. CSV: Network flow records (Benign, PortScan, SYN Flood, SSH BruteForce)
2. PCAP: Real packet captures using Scapy
3. Zeek: Tab-separated conn.log format

Ensures sufficient consecutive time windows (>= 12 windows) for temporal
sequence building and multi-step world model forecasting.
"""

from pathlib import Path
from datetime import datetime, timedelta
import random
import pandas as pd


def generate_sample_csv(output_path: Path, num_windows: int = 15):
    """
    Generate a realistic network flow CSV with multiple attack patterns
    across sequential time windows.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2026, 9, 8, 10, 0, 0)
    
    rows = []
    window_duration_sec = 5

    # Known IPs
    internal_clients = ["10.0.0.15", "10.0.0.16", "10.0.0.17"]
    servers = ["10.0.0.2", "10.0.0.5", "10.0.0.8"]
    external_attacker = "192.168.1.100"

    for w in range(num_windows):
        window_start = base_time + timedelta(seconds=w * window_duration_sec)
        
        # 1. Normal benign background traffic in every window
        for _ in range(random.randint(5, 8)):
            flow_time = window_start + timedelta(seconds=random.uniform(0.1, 4.8))
            client = random.choice(internal_clients)
            server = random.choice(servers)
            port = random.choice([80, 443, 53])
            proto = 6 if port in (80, 443) else 17
            fwd_pkts = random.randint(4, 25)
            bwd_pkts = random.randint(4, 30)
            fwd_bytes = fwd_pkts * random.randint(60, 300)
            bwd_bytes = bwd_pkts * random.randint(80, 1200)
            
            rows.append({
                "src_ip": client,
                "dst_ip": server,
                "src_port": random.randint(49152, 65535),
                "dst_port": port,
                "protocol": proto,
                "timestamp": flow_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": round(random.uniform(0.05, 2.5), 4),
                "fwd_packets": fwd_pkts,
                "bwd_packets": bwd_pkts,
                "fwd_bytes": fwd_bytes,
                "bwd_bytes": bwd_bytes,
                "syn_flags": 1 if proto == 6 else 0,
                "rst_flags": 0,
                "ack_flags": 1 if proto == 6 else 0,
                "label": "Benign"
            })
            
        # 2. Port scan behavior starting around window 4
        if 4 <= w <= 8:
            for target_port in [21, 22, 23, 25, 80, 443, 8080, 8443, 3389]:
                flow_time = window_start + timedelta(seconds=random.uniform(0.1, 4.8))
                rows.append({
                    "src_ip": external_attacker,
                    "dst_ip": servers[0],
                    "src_port": random.randint(50000, 60000),
                    "dst_port": target_port,
                    "protocol": 6,
                    "timestamp": flow_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": 0.05,
                    "fwd_packets": 1,
                    "bwd_packets": 0 if target_port not in (80, 443) else 1,
                    "fwd_bytes": 60,
                    "bwd_bytes": 0 if target_port not in (80, 443) else 60,
                    "syn_flags": 1,
                    "rst_flags": 1 if target_port not in (80, 443) else 0,
                    "ack_flags": 0,
                    "label": "PortScan"
                })

        # 3. High-volume SYN flood / brute force starting in window 9
        if w >= 9:
            for _ in range(15):
                flow_time = window_start + timedelta(seconds=random.uniform(0.05, 4.9))
                rows.append({
                    "src_ip": external_attacker,
                    "dst_ip": servers[0],
                    "src_port": random.randint(1024, 65535),
                    "dst_port": 22 if w % 2 == 0 else 80,
                    "protocol": 6,
                    "timestamp": flow_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": 0.01,
                    "fwd_packets": random.randint(5, 20),
                    "bwd_packets": 0,
                    "fwd_bytes": random.randint(300, 1200),
                    "bwd_bytes": 0,
                    "syn_flags": random.randint(5, 15),
                    "rst_flags": random.randint(0, 5),
                    "ack_flags": 0,
                    "label": "DoS-SynFlood" if w >= 11 else "SSH-Bruteforce"
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[+] Successfully generated CSV: {output_path} ({len(df)} records across {num_windows} windows)")
    return df


def generate_sample_pcap(output_path: Path, num_windows: int = 14):
    """
    Generate a valid PCAP file using Scapy.
    """
    try:
        from scapy.all import Ether, IP, TCP, UDP, wrpcap
    except ImportError:
        print("[-] Scapy not installed; skipping PCAP generation.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_epoch = 1757300000.0
    packets = []

    for w in range(num_windows):
        win_time = base_epoch + (w * 5.0)
        
        # Normal traffic
        for i in range(4):
            pkt_time = win_time + (i * 0.8)
            pkt = Ether() / IP(src="10.0.0.15", dst="10.0.0.2") / TCP(sport=51000 + i, dport=443, flags="PA")
            pkt.time = pkt_time
            packets.append(pkt)
            
        # Attack traffic in later windows
        if w >= 6:
            for p in range(6):
                pkt_time = win_time + (p * 0.5)
                pkt = Ether() / IP(src="192.168.1.100", dst="10.0.0.2") / TCP(sport=60000 + p, dport=22 + p, flags="S")
                pkt.time = pkt_time
                packets.append(pkt)

    wrpcap(str(output_path), packets)
    print(f"[+] Successfully generated PCAP: {output_path} ({len(packets)} packets)")


def generate_sample_zeek(output_path: Path, num_windows: int = 15):
    """
    Generate a standard Zeek TSV conn.log.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_ts = 1757300000.0
    
    header = [
        "#separator \\x09",
        "#set_separator\t,",
        "#empty_field\t(empty)",
        "#unset_field\t-",
        "#path\tconn",
        "#open\t2026-09-08-10-00-00",
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\thistory",
        "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\ttime\tcount\tcount\tstring\tbool\tbool\tcount\tcount\tcount\tcount\tcount\tstring"
    ]
    
    lines = list(header)
    uid_counter = 1

    for w in range(num_windows):
        win_time = base_ts + (w * 5.0)
        
        # Benign traffic
        for _ in range(3):
            ts = win_time + random.uniform(0.1, 4.5)
            uid = f"C{uid_counter:04d}"
            uid_counter += 1
            line = f"{ts:.6f}\t{uid}\t10.0.0.10\t{random.randint(50000, 60000)}\t10.0.0.2\t443\ttcp\t0.150000\t1200\t4500\tSF\tT\tT\t0\t10\t1600\t12\t5200\tShADadF"
            lines.append(line)
            
        # Reconnaissance / Attack in later windows
        if w >= 5:
            for p in [22, 23, 80, 445, 3389, 8080]:
                ts = win_time + random.uniform(0.1, 4.5)
                uid = f"C{uid_counter:04d}"
                uid_counter += 1
                state = "S0" if w < 10 else "REJ"
                history = "S" if state == "S0" else "Sr"
                line = f"{ts:.6f}\t{uid}\t192.168.1.100\t{random.randint(50000, 60000)}\t10.0.0.2\t{p}\ttcp\t0.020000\t60\t0\t{state}\tF\tT\t0\t1\t60\t0\t0\t{history}"
                lines.append(line)

    lines.append("#close\t2026-09-08-10-02-00\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Successfully generated Zeek log: {output_path} ({len(lines)} lines)")


def main():
    root = Path(__file__).resolve().parent
    sample_dir = root / "data" / "sample"
    
    print("=" * 60)
    print("ARJUN SYNTHETIC TELEMETRY GENERATOR")
    print("=" * 60)
    
    generate_sample_csv(sample_dir / "sample_network_flows.csv")
    generate_sample_pcap(sample_dir / "sample_traffic.pcap")
    generate_sample_zeek(sample_dir / "sample_zeek_conn.log")
    
    print("=" * 60)
    print("Sample telemetry generated successfully in data/sample/")
    print("=" * 60)


if __name__ == "__main__":
    main()
