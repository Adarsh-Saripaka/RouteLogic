# Smart Routing ML System

A machine learning-powered smart routing system for computer networks that dynamically selects optimal paths based on real-time network parameters and node trust scores.

## 🚀 Quick Start

**Run the interactive web demo:**
```bash
cd /home/yugpatel/CN_PROJECT
source .venv/bin/activate
python src/app.py
```
Then open `http://localhost:5000` in your browser.

## 📋 Basic Idea

This system simulates intelligent network routing where:

1. **Network Topology**: Users can create custom node-link topologies
2. **ML Prediction**: Decision Tree model predicts best paths using features like delay, packet loss, bandwidth, load, and trust scores
3. **Trust Dynamics**: Node reliability evolves based on packet delivery success/failure
4. **Three Strategies**:
   - **Cost-Only**: Traditional shortest path (minimize hops/delay)
   - **ML-Only**: Pure machine learning prediction
   - **Hybrid**: 60% ML + 40% cost optimization
5. **Real-time Updates**: Trust scores update after each simulation, affecting future routing decisions

## 🎯 Key Features

- Interactive network builder with drag-and-drop nodes
- Live visualization with trust-based node coloring (🟢 Green = Reliable, 🔴 Red = Unreliable)
- Packet simulation with realistic success rates
- Persistent trust history across sessions
- Comparative analysis of different routing strategies

## 🛠️ Tech Stack

- **Backend**: Flask, NetworkX, scikit-learn
- **Frontend**: HTML5 Canvas, JavaScript
- **ML**: Decision Tree Classifier
- **Data**: Pandas, NumPy

## 📊 How It Works

1. **Build Network**: Add nodes and connections
2. **Configure Simulation**: Select source/dest, packet count, strategy
3. **Run Prediction**: ML model evaluates all possible paths
4. **Simulate Traffic**: Packets routed with trust-based success probabilities
5. **Update Trust**: Node scores change based on performance
6. **Visualize Results**: See path highlighting and trust evolution

The system demonstrates how ML can improve network routing by learning from historical performance and adapting to changing network conditions.
- `docs/`: Documentation