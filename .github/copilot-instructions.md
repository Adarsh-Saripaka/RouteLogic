<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Smart Routing ML System - Development Guidelines

## Project Overview
This is a Python project for **machine learning-based smart routing in computer networks** with live path visualization and real-time ML predictions.

## Core Libraries & Technologies

| Component | Library | Purpose |
|-----------|---------|---------|
| Network Simulation | NetworkX | Graph-based network topology and path simulation |
| ML Model | scikit-learn | Decision Tree classifier for optimal path prediction |
| Data Processing | Pandas/NumPy | Feature engineering and data handling |
| Model Persistence | joblib | Save/load trained ML models |

## Network Parameters (Live Features)

| Parameter | Type | Range | Purpose |
|-----------|------|-------|---------|
| **Delay** | Float | ms | Link latency/propagation delay |
| **Packet Loss** | Float | 0-1 | Link reliability metric |
| **Bandwidth** | Float | Mbps | Link capacity |
| **Load** | Float | 0-1 | Current link utilization |
| **Trust Score** | Float | 0-1 | Node reliability history |

## Node & Path Management

| Element | Description | Live Feature |
|---------|-------------|--------------|
| **Nodes** | Network router/switch endpoints | Draw/add nodes interactively |
| **Paths** | Routes between nodes | Draw paths and visualize in real-time |
| **Links** | Connections between nodes | Update parameters dynamically |
| **Topology** | Complete network graph | Visualized network view |

## Trust Score System

- **Initialization**: 0.5 (neutral initial trust)
- **Success Update**: +0.1 per successful packet delivery
- **Failure Update**: -0.1 per failed packet delivery
- **Range**: Clamped to [0, 1]

## ML Prediction Pipeline (Live)

| Stage | Process | Output |
|-------|---------|--------|
| **Feature Collection** | Gather live delay, loss, bandwidth, load, trust_avg | Feature vector |
| **ML Model** | Decision Tree predicts optimal path | Path probability scores |
| **Live Prediction** | Real-time routing decision | Best path recommendation |
| **Feedback Loop** | Update trust scores based on results | Improved future predictions |

## Development Focus
- Implement live path drawing/visualization feature
- Real-time parameter monitoring and updates
- ML model prediction latency optimization
- Trust score dynamic recalculation based on live network metrics