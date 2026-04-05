# Smart Routing System Documentation

## Overview
This system implements a machine learning-based routing protocol that considers multiple factors including trust scores, delay, packet loss, bandwidth, and network load.

## Components

### Network Simulation
- Uses NetworkX to model the network as a graph
- Nodes represent routers with trust scores
- Edges represent connections with dynamic parameters

### Trust Mechanism
- Each node has a trust score between 0 and 1
- Updated based on packet delivery success/failure
- Influences routing decisions

### Machine Learning Model
- Decision Tree classifier trained on path features
- Predicts optimal paths based on real-time parameters
- Features: delay, loss, bandwidth, load, trust

### Routing Algorithm
- Finds multiple possible paths
- Uses ML model to score and select best path
- Falls back to trust-based selection if model unavailable

## Usage
Run `python src/main.py` to simulate the network and find optimal paths.