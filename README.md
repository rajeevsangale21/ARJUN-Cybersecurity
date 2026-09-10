# ARJUN — AI-Based Network Attack Forecasting

### From Reactive Attack Detection to Proactive, Explainable Cyber Defence

ARJUN is an AI-driven cybersecurity prototype designed to forecast the future progression of network attacks from network traffic and Zeek telemetry.

Instead of only answering:

> **"Is the network under attack right now?"**

ARJUN aims to answer:

> **"What is likely to happen next?"**

The system represents network activity as temporal network states and communication graphs, learns state-transition dynamics using a hybrid **Temporal GNN + LSTM World Model**, and performs multi-step future-state rollout to estimate future attack probability, risk, and MITRE ATT&CK progression.

---

## Problem Statement

Traditional Intrusion Detection Systems (IDS) are primarily reactive. They detect malicious activity after suspicious behaviour has already occurred.

This creates a major limitation:

- Current attacks can be detected.
- Suspicious traffic can be classified.
- But the **future progression of an attack is difficult to estimate**.

ARJUN addresses this by modelling the network as a sequence of evolving states:

```text
S(t-9) → S(t-8) → ... → S(t-1) → S(t)
                                      ↓
                              ARJUN World Model
                                      ↓
                    S(t+1) → S(t+2) → ... → S(t+K)