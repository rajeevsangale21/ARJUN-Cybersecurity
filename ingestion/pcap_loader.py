from pathlib import Path

import pandas as pd

from scapy.all import (
    rdpcap,
    IP,
    IPv6,
    TCP,
    UDP
)


def load_pcap(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"PCAP file not found: {path}"
        )


    packets = rdpcap(
        str(path)
    )


    rows = []


    for packet in packets:

        # ==========================================
        # IP layer
        # ==========================================

        if IP in packet:

            ip = packet[IP]

            src_ip = ip.src
            dst_ip = ip.dst

            ttl = int(
                ip.ttl
            )

            protocol = int(
                ip.proto
            )

            fragment_offset = int(
                ip.frag
            )

            more_fragments = bool(
                int(ip.flags) & 0x1
            )

        elif IPv6 in packet:

            ip = packet[IPv6]

            src_ip = ip.src
            dst_ip = ip.dst

            ttl = int(
                ip.hlim
            )

            protocol = int(
                ip.nh
            )

            fragment_offset = 0
            more_fragments = False

        else:

            continue


        # ==========================================
        # Transport layer
        # ==========================================

        src_port = None
        dst_port = None

        tcp_flags = ""

        tcp_window = None


        if TCP in packet:

            tcp = packet[TCP]

            src_port = int(
                tcp.sport
            )

            dst_port = int(
                tcp.dport
            )

            tcp_flags = str(
                tcp.flags
            )

            tcp_window = int(
                tcp.window
            )


        elif UDP in packet:

            udp = packet[UDP]

            src_port = int(
                udp.sport
            )

            dst_port = int(
                udp.dport
            )


        # ==========================================
        # Packet length
        # ==========================================

        packet_length = len(
            packet
        )


        # ==========================================
        # Actual payload size
        # ==========================================

        payload_size = 0

        if TCP in packet:

            payload_size = len(
                bytes(packet[TCP].payload)
            )

        elif UDP in packet:

            payload_size = len(
                bytes(packet[UDP].payload)
            )


        # ==========================================
        # Fragment flag
        # ==========================================

        fragment_flag = int(
            fragment_offset > 0
            or more_fragments
        )


        # ==========================================
        # Store packet
        # ==========================================

        rows.append(
            {
                "timestamp":
                    float(packet.time),

                "src_ip":
                    src_ip,

                "dst_ip":
                    dst_ip,

                "src_port":
                    src_port,

                "dst_port":
                    dst_port,

                "protocol":
                    protocol,

                "ttl":
                    ttl,

                "tcp_flags":
                    tcp_flags,

                "packet_length":
                    packet_length,

                "payload_size":
                    payload_size,

                "tcp_window":
                    tcp_window,

                "fragment_offset":
                    fragment_offset,

                "fragment_flag":
                    fragment_flag
            }
        )


    # ==============================================
    # Convert to DataFrame
    # ==============================================

    df = pd.DataFrame(
        rows
    )


    if df.empty:

        raise ValueError(
            "No IP packets were found "
            "in the PCAP file."
        )


    return df