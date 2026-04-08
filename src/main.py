"""
main.py — SmartRoutingNetwork core class
Fixes applied:
  1. _create_realistic_topology: grid edge logic was off-by-one for non-square counts
  2. get_path_features: node_positions may be a dict of np.ndarray — safe distance calc
  3. simulate_packet_forwarding: path[1:-1] skipped destination — corrected to path[1:]
  4. find_best_path: best_path initialised to None then indexed — now initialised to paths[0]
  5. train_model: X/y initialised to [] not None — avoids confusing branch logic
  6. generate_training_data: robust path selection when graph may be sparse
  7. calculate_qos_score: division-safe bandwidth normalisation (was /200, now /100)
  8. detect_anomalies: returns consistent dict shape matching app.py expectations
  9. All numpy int/float scalars cast explicitly before returning to avoid JSON issues
 10. Random seed management improved via np.random.default_rng for reproducibility
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib


class SmartRoutingNetwork:
    """
    Core network simulation class.
    Used standalone (python main.py) or imported by app.py for topology utilities.
    """

    def __init__(self, num_nodes: int = 10):
        self.num_nodes     = num_nodes
        self.graph         = self._create_realistic_topology(num_nodes)
        self.trust_scores  = {node: 0.5 for node in self.graph.nodes()}
        self.node_positions= self._calculate_positions()
        self.anomaly_scores= {node: 0.0 for node in self.graph.nodes()}
        # QoS class per node: 0=best_effort, 1=premium, 2=real_time
        rng = np.random.default_rng(0)
        self.node_qos_class = {
            node: int(rng.integers(0, 3)) for node in self.graph.nodes()
        }
        self.model      = None
        self.model_type = None
        self.last_score = None

    # ══════════════════════════════════════════════════════════════════
    #  TOPOLOGY
    # ══════════════════════════════════════════════════════════════════

    def _create_realistic_topology(self, num_nodes: int) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        # FIX: grid_size based on sqrt; correct column-wrap so last node of each
        # row doesn't connect to first node of the next row
        grid_size = max(1, int(np.ceil(np.sqrt(num_nodes))))
        for i in range(num_nodes):
            row, col = divmod(i, grid_size)
            # horizontal neighbour
            if col < grid_size - 1 and (i + 1) < num_nodes:
                G.add_edge(i, i + 1)
            # vertical neighbour
            if (i + grid_size) < num_nodes:
                G.add_edge(i, i + grid_size)

        # Random long-range edges for richer topology
        rng = np.random.default_rng(42)
        for _ in range(max(1, num_nodes // 3)):
            n1, n2 = rng.choice(num_nodes, 2, replace=False)
            n1, n2 = int(n1), int(n2)
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2)

        return G

    def _calculate_positions(self) -> dict:
        return dict(nx.spring_layout(self.graph, seed=42))

    # ══════════════════════════════════════════════════════════════════
    #  TRUST MANAGEMENT
    # ══════════════════════════════════════════════════════════════════

    def update_trust(self, node, success: bool,
                     packet_count: int = 1, congestion_factor: float = 1.0):
        current = self.trust_scores[node]
        if success:
            delta = 0.02 * packet_count * (1.0 - current) * congestion_factor
            self.trust_scores[node] = float(min(1.0, current + delta))
        else:
            delta = 0.08 * packet_count / max(congestion_factor, 0.01)
            self.trust_scores[node] = float(max(0.1, current - delta))

    def apply_trust_decay(self, decay_rate: float = 0.001):
        for node in self.trust_scores:
            self.trust_scores[node] = float(
                max(0.1, self.trust_scores[node] * (1.0 - decay_rate)))

    # ══════════════════════════════════════════════════════════════════
    #  ANOMALY DETECTION
    # ══════════════════════════════════════════════════════════════════

    def detect_anomalies(self, path: list, threshold: float = 0.3) -> dict:
        """
        Flag nodes whose trust deviates significantly below baseline (0.5).
        Returns a dict keyed by str(node) to match app.py / frontend expectations.
        """
        result = {}
        for node in path:
            trust     = float(self.trust_scores.get(node, 0.5))
            deviation = max(0.0, 0.5 - trust)
            score     = round(deviation * 2.0, 3)
            flagged   = score > threshold
            self.anomaly_scores[node] = score
            result[str(node)] = {
                'score':   score,
                'flagged': bool(flagged),
                'trust':   round(trust, 3),
                'fatigue': 0.0,          # standalone mode has no fatigue tracker
                'reason':  'Trust below safe threshold' if flagged else 'Operating normally',
            }
        return result

    # ══════════════════════════════════════════════════════════════════
    #  QoS SCORING
    # ══════════════════════════════════════════════════════════════════

    def calculate_qos_score(self, path: list, priority: str = 'best_effort') -> float:
        """Score a path according to the given QoS service class."""
        qos_weights = {
            'real_time':   {'delay': 0.6,  'loss': 0.3,  'bandwidth': 0.05, 'load': 0.05},
            'premium':     {'delay': 0.3,  'loss': 0.3,  'bandwidth': 0.2,  'load': 0.2 },
            'best_effort': {'delay': 0.2,  'loss': 0.2,  'bandwidth': 0.3,  'load': 0.3 },
        }
        W        = qos_weights.get(priority, qos_weights['best_effort'])
        features = self.get_path_features(path)
        delay, loss, bandwidth, load, _trust = features
        score = (
            W['delay']     * max(0.0, 1.0 / (1.0 + delay * 0.1)) +
            W['loss']      * max(0.0, 1.0 - loss) +
            # FIX: normalise by 100 Mbps ceiling, not 200
            W['bandwidth'] * min(1.0, bandwidth / 100.0) +
            W['load']      * max(0.0, 1.0 - load)
        )
        return round(float(score), 4)

    # ══════════════════════════════════════════════════════════════════
    #  PACKET SIMULATION
    # ══════════════════════════════════════════════════════════════════

    def simulate_packet_forwarding(self, path: list, num_packets: int = 10) -> int:
        """
        Simulate packet delivery along path.
        FIX: original used path[1:-1] which skipped the destination hop.
             Correct iteration is path[1:].
        """
        successful = 0
        for _ in range(num_packets):
            packet_ok   = True
            congestion  = self._calculate_path_congestion(path)
            for i, node in enumerate(path[1:]):     # FIX: was path[1:-1]
                hop_cong  = congestion * (1.0 + 0.1 * i)
                base_prob = self.trust_scores.get(node, 0.5)
                cong_pen  = min(0.5, hop_cong)
                final_prob= base_prob * (1.0 - cong_pen)
                ok        = np.random.random() < max(0.0, final_prob)
                self.update_trust(node, ok, 1,
                                  1.0 / hop_cong if ok else hop_cong)
                if not ok:
                    packet_ok = False
                    break
            if packet_ok:
                successful += 1
        self.apply_trust_decay()
        return successful

    def _calculate_path_congestion(self, path: list) -> float:
        avg_trust = float(np.mean([self.trust_scores.get(n, 0.5) for n in path]))
        return float(min(2.0, (len(path) / 10.0) * (1.0 / max(avg_trust, 0.01)) * 0.1))

    # ══════════════════════════════════════════════════════════════════
    #  PATH FEATURES
    # ══════════════════════════════════════════════════════════════════

    def get_path_features(self, path: list) -> list:
        """Return [avg_delay, avg_loss, avg_bandwidth, avg_load, avg_trust]."""
        if len(path) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.5]

        delays, losses, bandwidths, loads, trusts = [], [], [], [], []
        positions = self.node_positions or {}

        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]

            # FIX: node_positions values can be np.ndarray; use np.linalg.norm safely
            if n1 in positions and n2 in positions:
                p1  = np.asarray(positions[n1], dtype=float)
                p2  = np.asarray(positions[n2], dtype=float)
                dist = float(np.linalg.norm(p1 - p2))
                delay = dist * 2.0 + float(np.random.uniform(0.5, 2.0))
            else:
                delay = float(np.random.uniform(1.0, 10.0))

            trust_n1  = float(self.trust_scores.get(n1, 0.5))
            base_loss = 0.01
            trust_pen = (1.0 - trust_n1) * 0.05
            loss      = float(min(0.15, base_loss + trust_pen + np.random.uniform(0, 0.02)))

            edge = (n1, n2)
            bw   = (float(np.random.uniform(80, 200))
                    if self._is_backbone_edge(edge)
                    else float(np.random.uniform(10, 50)))
            load = float(min(1.0, (1.0 - trust_n1) * 0.8 + np.random.uniform(0, 0.3)))

            delays.append(delay)
            losses.append(loss)
            bandwidths.append(bw)
            loads.append(load)
            trusts.append(trust_n1)

        return [
            float(np.mean(delays)),
            float(np.mean(losses)),
            float(np.mean(bandwidths)),
            float(np.mean(loads)),
            float(np.mean(trusts)),
        ]

    def _is_backbone_edge(self, edge: tuple) -> bool:
        n1, n2 = edge
        positions = self.node_positions or {}
        if n1 in positions and n2 in positions:
            p1   = np.asarray(positions[n1], dtype=float)
            p2   = np.asarray(positions[n2], dtype=float)
            return float(np.linalg.norm(p1 - p2)) > 1.5
        return False

    # ══════════════════════════════════════════════════════════════════
    #  TRAINING DATA GENERATION
    # ══════════════════════════════════════════════════════════════════

    def generate_training_data(self, num_samples: int = 500):
        all_paths = []
        for src in self.graph.nodes():
            for dst in self.graph.nodes():
                if src >= dst:
                    continue
                # FIX: only enumerate paths for connected node pairs
                if nx.has_path(self.graph, src, dst):
                    all_paths.extend(
                        nx.all_simple_paths(self.graph, src, dst, cutoff=5))

        # Fallback: at least one path
        if not all_paths:
            nodes = list(self.graph.nodes())
            all_paths = [[nodes[0], nodes[-1]]] if len(nodes) >= 2 else [[0, 1]]

        rng     = np.random.default_rng(0)
        samples, labels = [], []

        for _ in range(num_samples):
            idx     = int(rng.integers(0, len(all_paths)))
            path    = all_paths[idx]
            features= self.get_path_features(path)
            delay, loss, bandwidth, load, trust = features
            score   = (
                0.25 * (1.0 / (1.0 + delay))
              + 0.25 * (1.0 - loss)
              + 0.20 * min(1.0, bandwidth / 100.0)
              + 0.20 * (1.0 - load)
              + 0.10 * trust
            )
            samples.append(features)
            labels.append(1 if score > 0.65 else 0)

        return np.array(samples), np.array(labels)

    # ══════════════════════════════════════════════════════════════════
    #  MODEL TRAINING
    # ══════════════════════════════════════════════════════════════════

    def train_model(self, model_type: str = 'random_forest') -> dict | None:
        data_dir   = os.path.join(os.path.dirname(__file__), '..', '.data')
        models_dir = os.path.join(os.path.dirname(__file__), '..', '.models')

        # FIX: initialise X, y as empty lists (not None) to simplify branching
        X, y = [], []
        history_path = os.path.join(data_dir, 'routing_history.csv')
        if os.path.exists(history_path):
            try:
                df = pd.read_csv(history_path).dropna(
                    subset=['avg_delay', 'packet_loss', 'bandwidth', 'load', 'trust_avg'])
                if len(df) >= 10:
                    X = df[['avg_delay', 'packet_loss', 'bandwidth',
                             'load', 'trust_avg']].values.tolist()
                    y = (df['success_rate'] > 0.5).astype(int).values.tolist()
            except Exception as e:
                print(f"⚠️  Could not read CSV: {e}")

        if len(X) < 100:
            print("Generating synthetic training data…")
            X_gen, y_gen = self.generate_training_data(1000)
            X.extend(X_gen.tolist())
            y.extend(y_gen.tolist())

        if not X:
            print("⚠️  No training data available.")
            return None

        # Build model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            self.model = DecisionTreeClassifier(random_state=42)

        X_arr = np.array(X)
        y_arr = np.array(y)
        self.model.fit(X_arr, y_arr)

        score = None
        if len(X_arr) > 50:
            from sklearn.model_selection import train_test_split
            Xtr, Xte, ytr, yte = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=42)
            self.model.fit(Xtr, ytr)
            score = float(self.model.score(Xte, yte))
            print(f"✅ {model_type} accuracy: {score:.3f}")
        else:
            print(f"✅ {model_type} trained on {len(X_arr)} samples")

        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f'{model_type}.pkl')
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved: {model_path}")

        self.model_type = model_type
        self.last_score = score
        return {
            'model':    model_type,
            'samples':  len(X_arr),
            'accuracy': score,
        }

    # ══════════════════════════════════════════════════════════════════
    #  PATH SELECTION
    # ══════════════════════════════════════════════════════════════════

    def calculate_cost(self, features: list) -> float:
        delay, loss, bandwidth, load, trust = features
        cost = (delay * 0.4
                + loss * 100.0 * 0.3
                + (1.0 - min(1.0, bandwidth / 100.0)) * 0.1
                + load * 0.2)
        trust_penalty = (1.0 - trust) * 2.0
        return float(cost + trust_penalty)

    def find_best_path(self, source: int, dest: int,
                       strategy: str = 'hybrid') -> list | None:
        
        # Exclude completely blocked/compromised nodes
        blocked_nodes = [
            n for n in self.graph.nodes() 
            if self.trust_scores.get(n, 0.5) <= 0.1
        ]
        
        active_graph = self.graph.copy()
        active_graph.remove_nodes_from(blocked_nodes)
        
        if source in blocked_nodes or dest in blocked_nodes:
            print(f"⚠️  Source or destination node is compromised.")
            return None

        if not nx.has_path(active_graph, source, dest):
            print(f"⚠️  No path between {source} and {dest} (network partitioned).")
            return None

        paths = list(nx.all_simple_paths(active_graph, source, dest, cutoff=5))
        if not paths:
            return None

        # FIX: initialise best_path to first candidate so we always return a valid path
        best_path  = paths[0]
        best_score = -float('inf')

        for path in paths:
            features   = self.get_path_features(path)
            cost       = self.calculate_cost(features)
            model_score = 0.0

            if self.model is not None:
                try:
                    proba       = self.model.predict_proba([features])[0]
                    model_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                except Exception:
                    model_score = float(self.model.predict([features])[0])

            if strategy == 'ml':
                score = model_score
            elif strategy == 'cost':
                score = 1.0 / (1.0 + cost)
            else:   # hybrid
                score = 0.6 * model_score + 0.4 * (1.0 / (1.0 + cost))

            if score > best_score:
                best_score = score
                best_path  = path

        return best_path


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  Smart Routing Network — Standalone Test")
    print("=" * 60)

    network = SmartRoutingNetwork(num_nodes=10)
    print(f"\nNodes : {list(network.graph.nodes())}")
    print(f"Edges : {list(network.graph.edges())}")
    print(f"\nInitial trust scores:")
    for n, t in network.trust_scores.items():
        print(f"  Node {n}: {t:.3f}")

    print("\n🧠 Training model (random_forest)…")
    result = network.train_model('random_forest')
    if result:
        acc = result.get('accuracy')
        print(f"   Samples: {result['samples']}  |  Accuracy: {acc:.3f}" if acc else
              f"   Samples: {result['samples']}")

    print("\n🔍 Finding best path: Node 0 → Node 9")
    path = network.find_best_path(0, 9)
    if path:
        print(f"   Path: {' → '.join(map(str, path))}")
        print("\n📦 Simulating 20 packets…")
        ok = network.simulate_packet_forwarding(path, num_packets=20)
        print(f"   Delivered: {ok}/20 ({ok/20*100:.1f}%)")

        print("\n📊 QoS scores:")
        for cls in ('best_effort', 'premium', 'real_time'):
            print(f"   {cls:12s}: {network.calculate_qos_score(path, cls):.4f}")

        print("\n🛡️  Anomaly detection:")
        anomalies = network.detect_anomalies(path)
        for node, info in anomalies.items():
            flag = "🚨 FLAGGED" if info['flagged'] else "✅ OK"
            print(f"   Node {node}: score={info['score']:.3f}  trust={info['trust']:.3f}  {flag}")

        print("\nFinal trust scores:")
        for n, t in network.trust_scores.items():
            print(f"  Node {n}: {t:.4f}")
    else:
        print("   ⚠️  No path found between 0 and 9")

    print("\n✅ Done.")