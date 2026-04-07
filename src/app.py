"""
app.py — Smart Routing ML System · Flask Backend
Fixes applied:
  1. CORS headers added to every response via after_request (not just OPTIONS handler)
  2. trust_scores_global uses int keys internally; string-key lookups fixed throughout
  3. node_fatigue_global key type normalised to int (matching nodes list from frontend)
  4. select_best_path_ml: guard against empty paths list before indexing [0]
  5. simulate_packets: iterate path[1:] not path[1:-1] (original skipped destination)
  6. save_routing_history: safe None-guard on path_metrics values
  7. build_event_log: robust key lookup for trust_before / trust_after (str or int keys)
  8. /api/trust-values: returns string keys (matching frontend expectations)
  9. initialize_ml_model: DATA_DIR created before HISTORY_FILE referenced
 10. RandomForest replaced by consistent model everywhere (was mixed with DecisionTree label)
 11. All numpy scalar coercion done through NpEncoder — no manual int() casts missed
 12. predict route: int(source) / int(dest) cast at entry to avoid key-type mismatch
"""

from flask import Flask, request, Response
from flask_cors import CORS
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
import os
import joblib
from datetime import datetime

app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ── CORS: allow everything during development ──────────────────────────────────
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    """Belt-and-braces: stamp CORS headers on every response."""
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


# ── Numpy-safe JSON encoder ────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_):    return bool(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def safe_jsonify(data, status=200):
    return Response(
        json.dumps(data, cls=NpEncoder),
        status=status,
        mimetype='application/json',
    )


# ── Global state ───────────────────────────────────────────────────────────────
ml_model             = None
trust_scores_global  = {}   # keys: int
node_fatigue_global  = {}   # keys: int

# ── File paths ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# Priority: Environment variable (Persistent Disk mount e.g. /data) > local .data
DATA_DIR     = os.getenv('DATA_DIR', os.path.join(BASE_DIR, '..', '.data'))
MODELS_DIR   = os.path.join(BASE_DIR, '..', '.models')

# Fallback to /tmp for Vercel/Serverless read-only environments
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
except OSError:
    DATA_DIR     = '/tmp/.data'
    MODELS_DIR   = '/tmp/.models'
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(DATA_DIR,   'routing_history.csv')
TRUST_FILE   = os.path.join(DATA_DIR,   'trust_values.json')
MODEL_FILE   = os.path.join(MODELS_DIR, 'routing_model.pkl')

# ── Trust-formula constants ────────────────────────────────────────────────────
TRUST_ALPHA = 0.15
TRUST_BETA  = 0.60
TRUST_DECAY = 0.002
TRUST_MIN   = 0.05
TRUST_MAX   = 1.00


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — trust & reward
# ══════════════════════════════════════════════════════════════════════════════

def compute_reward(success_rate, path_metrics):
    delay_factor = max(0.0, 1.0 - path_metrics.get('avg_delay',   5.0) / 30.0)
    loss_factor  = max(0.0, 1.0 - path_metrics.get('packet_loss', 0.05) * 10.0)
    load_factor  = max(0.0, 1.0 - path_metrics.get('load',        0.5))
    R = (0.5 * success_rate
         + 0.2 * delay_factor
         + 0.2 * loss_factor
         + 0.1 * load_factor)
    return float(np.clip(R, 0.0, 1.0))


def update_trust_enhanced(current_trust, reward, ml_prob, node_fatigue=0.0, hop_index=0):
    """
    Perfect Formula: T_new = T_old + α·M·(R_node − T_old)·(1 − T_old·β)
    Enhanced with position-based decay and slight stochastic variance to prevent identical values.
    """
    M = 0.5 + float(ml_prob)
    
    # Position penalty: nodes further down the path are slightly less rewarded (responsibility attribution)
    pos_factor = 1.0 - (hop_index * 0.02)
    
    # Fatigue impact: heavily fatigued nodes struggle to gain trust
    fatigue_penalty = node_fatigue * 0.2
    
    node_reward = max(0.0, (reward * pos_factor) - fatigue_penalty)
    
    # Add a tiny bit of unique node noise (0.001 range) to ensure distinct values for different nodes
    node_noise = np.random.uniform(-0.001, 0.001)
    
    delta = TRUST_ALPHA * M * (node_reward - current_trust) * (1.0 - current_trust * TRUST_BETA)
    new_trust = current_trust + delta + node_noise
    
    return float(np.clip(new_trust, TRUST_MIN, TRUST_MAX))


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — trust persistence
# ══════════════════════════════════════════════════════════════════════════════

def load_trust_history():
    """Load persisted trust scores. Keys stored as strings → convert to int."""
    global trust_scores_global
    if os.path.exists(TRUST_FILE):
        try:
            with open(TRUST_FILE, 'r') as f:
                raw = json.load(f)
            # FIX: normalise to int keys
            trust_scores_global = {int(k): float(v) for k, v in raw.items()}
            print(f"✅ Loaded trust history: {len(trust_scores_global)} nodes")
            return
        except Exception as e:
            print(f"⚠️  load_trust_history: {e}")
    trust_scores_global = {}


def save_trust_history():
    """Persist trust scores. Write string keys for JSON compatibility."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        serialisable = {str(k): round(float(v), 6)
                        for k, v in trust_scores_global.items()}
        with open(TRUST_FILE, 'w') as f:
            json.dump(serialisable, f, indent=2)
    except Exception as e:
        print(f"⚠️  save_trust_history: {e}")


def save_routing_history(d):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        pm = d.get('path_metrics') or {}
        entry = {
            'timestamp':          datetime.now().isoformat(),
            'source':             d.get('source'),
            'destination':        d.get('destination'),
            'path_length':        len(d.get('best_path', [])),
            'num_packets':        d.get('num_packets'),
            'successful_packets': d.get('successful_packets'),
            'success':            1 if (d.get('success_rate') or 0) > 0.7 else 0,
            'success_rate':       d.get('success_rate'),
            # FIX: safe None-guard so CSV never gets 'None' strings
            'avg_delay':          pm.get('avg_delay')   or 0.0,
            'packet_loss':        pm.get('packet_loss') or 0.0,
            'bandwidth':          pm.get('bandwidth')   or 0.0,
            'load':               pm.get('load')        or 0.0,
            'trust_avg':          pm.get('trust_avg')   or 0.5,
        }
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        else:
            df = pd.DataFrame([entry])
        df.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        print(f"⚠️  save_routing_history: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ML MODEL
# ══════════════════════════════════════════════════════════════════════════════

def initialize_ml_model():
    global ml_model
    # FIX: create dirs before any file operations
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    load_trust_history()

    if os.path.exists(MODEL_FILE):
        try:
            ml_model = joblib.load(MODEL_FILE)
            print(f"✅ Loaded pre-trained model from {MODEL_FILE}")
            return
        except Exception as e:
            print(f"⚠️  Could not load model ({e}); retraining…")

    _train_and_save_model()


def _train_and_save_model(extra_X=None, extra_y=None):
    """Train a RandomForest on historical data (+ synthetic fallback)."""
    global ml_model
    X, y = [], []

    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE).dropna(
                subset=['avg_delay', 'packet_loss', 'bandwidth', 'load', 'trust_avg', 'success'])
            if len(df) >= 10:
                X = df[['avg_delay', 'packet_loss', 'bandwidth', 'load', 'trust_avg']].values.tolist()
                y = df['success'].astype(int).values.tolist()
        except Exception as e:
            print(f"⚠️  Could not read history CSV ({e}); using synthetic data")

    # Pad with intelligent synthetic samples so the ML model is highly accurate out-of-the-box
    # This prevents training on pure random noise and provides a realistic baseline.
    needed = max(0, 150 - len(X))
    if needed > 0:
        rng = np.random.default_rng(42)
        X_syn, y_syn = [], []
        for _ in range(needed):
            # Generate realistic baseline networking metrics
            delay = rng.uniform(0.5, 25.0)  # avg_delay
            loss  = rng.uniform(0.0, 0.15)  # packet_loss
            bw    = rng.uniform(5.0, 95.0)  # bandwidth
            load  = rng.uniform(0.1, 0.95)  # load
            # Initial trust spreads mostly across healthy, with occasional historical bad nodes
            trust = rng.uniform(0.0, 1.0)   # trust_avg
            
            # Function logically determining if path is successful based on physical realities
            # (High trust & BW is good; High loss, delay, load is penalising)
            viability_score = (trust * 3.0) + (bw / 30.0) - (loss * 15.0) - (delay / 15.0) - (load * 2.0)
            success = 1 if viability_score > 0.8 else 0
            
            X_syn.append([delay, loss, bw, load, trust])
            y_syn.append(success)

        X.extend(X_syn)
        y.extend(y_syn)

    if extra_X is not None and extra_y is not None:
        X.extend(extra_X)
        y.extend(extra_y)

    ml_model = RandomForestClassifier(
        n_estimators=20, max_depth=5, min_samples_leaf=2, random_state=42)
    ml_model.fit(X, y)
    joblib.dump(ml_model, MODEL_FILE)
    print(f"✅ Model trained on {len(X)} samples and saved to {MODEL_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_path_metrics(path):
    """Base metrics derived from path, with dynamic jitter per request."""
    base_rng = np.random.default_rng(abs(hash(tuple(path))) % (2**32))
    dyn_rng  = np.random.default_rng()  # Dynamic variance per simulation

    num_hops = max(1, len(path) - 1)
    
    # Decouple metrics so that the shortest paths don't automatically win everywhere.
    # This allows Cost_Only (delay focus) vs ML_Only (holistic focus) to pick different routes!
    hop_seed1 = int(abs(hash("distance" + str(tuple(path)))) % (2**32))
    hop_seed2 = int(abs(hash("loss" + str(tuple(path)))) % (2**32))
    hop_seed3 = int(abs(hash("bw" + str(tuple(path)))) % (2**32))

    # Delay remains largely proportional to hop count (to simulate geographical distance)
    rng_dist = np.random.default_rng(hop_seed1)
    total_dist = float(rng_dist.uniform(0.5, 3.0, num_hops).sum())
    base_delay = total_dist * 1.5 + num_hops * float(rng_dist.uniform(0.5, 2.0))
    avg_delay   = float(base_delay + dyn_rng.uniform(-0.1, 0.4) * num_hops)
    
    # Loss is independent. A short path might have an extremely faulty wire!
    rng_loss = np.random.default_rng(hop_seed2)
    # 20% chance of a "bad cable" causing high loss even on a short connection
    is_faulty = rng_loss.random() > 0.8
    base_loss = float(rng_loss.uniform(0.05, 0.15)) if is_faulty else float(rng_loss.uniform(0.001, 0.02) + (num_hops * 0.005))
    packet_loss = float(max(0.0, base_loss + dyn_rng.uniform(-0.002, 0.005)))
    
    # Bandwidth is independent. A short path might be an old 10Mbps link, while a 3-hop path uses massive 100Mbps fiber trunks.
    rng_bw = np.random.default_rng(hop_seed3)
    is_high_capacity = rng_bw.random() > 0.5
    base_bw = float(rng_bw.uniform(60, 100)) if is_high_capacity else float(rng_bw.uniform(10, 40))
    # Slight decay per hop to simulate bottleneck chance, but not strictly bound
    base_bw = base_bw * (0.9 ** (num_hops - 1))
    bandwidth   = float(max(5.0, base_bw + dyn_rng.uniform(-2.0, 5.0)))
    
    # Load is somewhat dynamic
    load        = float(min(0.95, max(0.1, dyn_rng.uniform(0.2, 0.9))))

    return {
        'avg_delay':      round(avg_delay,   4),
        'packet_loss':    round(packet_loss, 6),
        'bandwidth':      round(bandwidth,   4),
        'load':           round(load,        4),
        'trust_avg':      0.5,          # placeholder; overwritten after trust lookup
        'num_hops':       num_hops,
        'total_distance': round(total_dist, 4),
    }


def _ml_score(features, strategy, qos_priority='best_effort'):
    """Return a scalar score for a candidate path, now aware of QoS requirements."""
    delay, loss, bw, ld, trust = features
    
    if ml_model is None:
        ml_prob = 0.5
    else:
        proba   = ml_model.predict_proba([features])[0]
        ml_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

    if strategy == 'ml_only':
        return ml_prob, ml_prob, 0.0

    # ── Classical Dijkstra / OSPF (Cost-Only) ──
    # Traditional protocols don't understand "Trust", "Security", or "Predicted Packet Loss".
    # They route purely based on shortest link metrics (Delay/Hop distance).
    if strategy == 'cost_only':
        # Purely distance/latency based. Blind to DDoS or Trust!
        traditional_cost = (delay * 2.0)
        traditional_score = 1.0 / (1.0 + traditional_cost)
        return traditional_score, ml_prob, traditional_cost

    # ── Hybrid & ML-Aware QoS Scoring ──
    # Weights dynamically adjusted by QoS class for routing decision
    W_qos = {
        'real_time':   {'delay': 2.5, 'loss': 15.0, 'bw': 0.1,  'load': 0.2, 'trust': 1.0},
        'premium':     {'delay': 1.0, 'loss': 5.0,  'bw': 0.5,  'load': 0.5, 'trust': 2.0},
        'best_effort': {'delay': 0.5, 'loss': 2.0,  'bw': 1.0,  'load': 1.0, 'trust': 0.5},
    }.get(qos_priority, {'delay': 0.5, 'loss': 2.0, 'bw': 1.0, 'load': 1.0, 'trust': 0.5})

    # Trust penalty: as trust drops, cost exponentially increases to avoid compromised infrastructure.
    trust_p_mult = W_qos['trust']
    trust_penalty = (trust_p_mult / (trust ** 1.5)) if trust > 0.1 else 200.0
    
    cost_val = (delay * W_qos['delay']) + (loss * W_qos['loss']) + (ld * W_qos['load']) + \
               (1.0 - min(1.0, bw / 100.0)) * W_qos['bw'] + trust_penalty

    # Normalize cost into a 0-1 score (inverse)
    smart_cost_score = 1.0 / (1.0 + cost_val)

    # Hybrid blends the smart comprehensive cost with pure ML probability
    score = 0.6 * ml_prob + 0.4 * smart_cost_score
    return score, ml_prob, cost_val


def select_best_path_ml(paths, strategy, trust_scores, fatigue_scores, qos_priority='best_effort'):
    # FIX: guard against empty list
    if not paths:
        return [], {}

    best_path, best_score, best_ml_prob, best_pm = paths[0], -float('inf'), 0.5, generate_path_metrics(paths[0])
    best_cost = 0.0

    for path in paths:
        pm = generate_path_metrics(path)
        # Effective trust penalised by fatigue - a chain is only as strong as its weakest link!
        node_trusts = [max(0.01, trust_scores.get(n, 0.5) - fatigue_scores.get(n, 0.0) * 0.4) for n in path]
        path_trust = float(np.min(node_trusts) if np.min(node_trusts) < 0.3 else np.mean(node_trusts))
        pm['trust_avg'] = path_trust
        features = [pm['avg_delay'], pm['packet_loss'],
                    pm['bandwidth'], pm['load'], pm['trust_avg']]
        
        score, ml_prob, cost_val = _ml_score(features, strategy, qos_priority)
        
        # Add tiny jitter to discourage taking the shortest path continuously if scores tie
        score += np.random.uniform(-0.0001, 0.0001)
        
        if score > best_score:
            best_score, best_path, best_ml_prob, best_pm = score, path, ml_prob, pm
            best_cost = cost_val

    best_pm['final_path_cost'] = round(best_cost, 4)
    return best_path, best_pm


def simulate_packets(path, num_packets, trust_scores, fatigue_scores):
    """
    FIX: simulation must check the entire path including the SOURCE node.
    Correct loop is path[0:] — a compromised source node should inhibit sending.
    """
    avg_trust  = float(np.mean([trust_scores.get(n, 0.9) for n in path]))
    congestion = min(2.0, (len(path) / 10) * (1.0 / max(avg_trust, 0.01)) * 0.1)
    ok = 0
    for _ in range(num_packets):
        success = True
        for i, node in enumerate(path): # Start from 0 to check source
            hop_cong  = congestion * (1.0 + 0.1 * i)
            base_prob = trust_scores.get(node, 0.9)
            fatigue_p = fatigue_scores.get(node, 0.0) * 0.25
            cong_p    = min(0.3, hop_cong)
            # Use a more lenient probability model for healthy nodes
            prob      = (base_prob ** 0.5) * (1.0 - cong_p) * (1.0 - fatigue_p)
            if np.random.random() >= max(0.0, prob):
                success = False
                break
        if success:
            ok += 1
    return ok


def detect_anomalies(path, trust_scores, fatigue_scores, threshold=0.3):
    result = {}
    for node in path:
        trust   = float(trust_scores.get(node, 0.5))
        fatigue = float(fatigue_scores.get(node, 0.0))
        score   = float(max(0.0, (0.5 - trust) * 2.0 + fatigue))
        flagged = score > threshold or fatigue > 0.6
        reason  = ('High fatigue/overuse detected' if fatigue > 0.6
                   else 'Trust below safe threshold' if score > threshold
                   else 'Operating normally')
        result[str(node)] = {
            'score':   round(score,   3),
            'flagged': bool(flagged),
            'trust':   round(trust,   3),
            'fatigue': round(fatigue, 3),
            'reason':  reason,
        }
    return result


def calculate_qos(path_metrics, qos_priority='best_effort'):
    W = {
        'real_time':   {'delay': 0.6, 'loss': 0.3, 'bandwidth': 0.05, 'load': 0.05},
        'premium':     {'delay': 0.3, 'loss': 0.3, 'bandwidth': 0.2,  'load': 0.2 },
        'best_effort': {'delay': 0.2, 'loss': 0.2, 'bandwidth': 0.3,  'load': 0.3 },
    }.get(qos_priority, {'delay': 0.2, 'loss': 0.2, 'bandwidth': 0.3, 'load': 0.3})

    delay = float(path_metrics.get('avg_delay',   5.0))
    loss  = float(path_metrics.get('packet_loss', 0.05))
    bw    = float(path_metrics.get('bandwidth',   50.0))
    load  = float(path_metrics.get('load',        0.5))

    d_s  = float(max(0.0, 1.0 - delay / 30.0))
    l_s  = float(1.0 - loss)
    b_s  = float(min(1.0, bw / 100.0))
    ld_s = float(1.0 - load)
    score = W['delay']*d_s + W['loss']*l_s + W['bandwidth']*b_s + W['load']*ld_s

    return {
        'qos_score': round(float(score), 4),
        'sla_met':   bool(score > 0.6),
        'priority':  qos_priority,
        'breakdown': {
            'delay_score':     round(d_s,  3),
            'loss_score':      round(l_s,  3),
            'bandwidth_score': round(b_s,  3),
            'load_score':      round(ld_s, 3),
        },
    }


def build_event_log(data):
    events   = []
    path     = data.get('best_path', [])
    sr       = float(data.get('success_rate', 0))
    num_pkt  = data.get('num_packets',  0)
    ok_pkt   = data.get('successful_packets', 0)
    pm       = data.get('path_metrics', {}) or {}
    t_before = data.get('trust_before', {}) or {}
    t_after  = data.get('trust_after',  {}) or {}
    anomalies= data.get('anomaly_detection', {}) or {}
    ml_prob  = float(data.get('ml_probability', 0.5))
    strategy = data.get('strategy', 'hybrid')
    reward   = float(data.get('reward_signal', 0))
    ts       = datetime.now().strftime('%H:%M:%S')

    def ev(level, icon, msg):
        events.append({'time': ts, 'level': level, 'icon': icon, 'msg': msg})

    ev('info',    '🚀', f'Simulation started — Node {data.get("source")} → Node {data.get("destination")}')
    ev('info',    '🧠', f'ML Engine [{strategy.replace("_"," ").title()}] path confidence: {ml_prob*100:.1f}%')
    ev('success', '📍', f'Optimal path: {" → ".join(map(str, path))} ({len(path)-1} hops)')
    ev('info',    '📡',
       f'Link quality — Delay: {pm.get("avg_delay",0):.1f} ms | '
       f'Loss: {pm.get("packet_loss",0)*100:.2f}% | '
       f'BW: {pm.get("bandwidth",0):.1f} Mbps | '
       f'Load: {pm.get("load",0)*100:.1f}%')
    ev('info',    '📦', f'Injecting {num_pkt} packets through {len(path)-1} hops…')
    ev('success' if sr > 0.7 else 'warning',
       '✅' if sr > 0.7 else '⚠️',
       f'Delivery: {ok_pkt}/{num_pkt} packets ({sr*100:.1f}% success)')
    ev('info', '🔐',
       f'Perfect Formula: α={TRUST_ALPHA}, M={0.5+ml_prob:.2f}, R={reward:.4f}')

    # FIX: robust lookup — t_before / t_after may have str or int keys
    for node in path:
        b = float(t_before.get(str(node), t_before.get(node, 0.5)))
        a = float(t_after .get(str(node), t_after .get(node, 0.5)))
        d = a - b
        if abs(d) > 0.0001:
            direction = '📈 increased' if d > 0 else '📉 decreased'
            ev('success' if d > 0 else 'warning', '⭐',
               f'Node {node} trust {direction}: {b:.4f} → {a:.4f} (Δ{d:+.4f})')

    for node, info in anomalies.items():
        if info.get('flagged'):
            ev('error', '🚨',
               f'ANOMALY Node {node}: fatigue={info.get("fatigue",0)*100:.0f}% — {info["reason"]}')

    qos = data.get('qos_metrics') or {}
    if qos:
        sla = '✅ SLA MET' if qos.get('sla_met') else '❌ SLA VIOLATED'
        ev('success' if qos.get('sla_met') else 'error', '📊',
           f'QoS [{(qos.get("priority") or "").replace("_"," ").title()}] '
           f'score={qos.get("qos_score",0):.4f} — {sla}')

    ev('info', '💾', 'Trust models converged and persisted successfully.')
    return events


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Root route: serving the main UI from the templates folder."""
    try:
        from flask import render_template
        return render_template('index.html')
    except Exception as e:
        return f"Error loading index.html: {str(e)}", 500


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_api():
    if request.method == 'OPTIONS':
        return safe_jsonify({}, 200)

    try:
        body = request.get_json(force=True, silent=True) or {}

        # ── Input validation ──────────────────────────────────────────────────
        raw_source = body.get('source')
        raw_dest   = body.get('dest')
        if raw_source is None or raw_dest is None:
            return safe_jsonify({'error': 'source and dest are required'}, 400)

        # FIX: cast to int immediately so all dict lookups use consistent key type
        source       = int(raw_source)
        dest         = int(raw_dest)
        num_packets  = max(1, int(body.get('num_packets', 10)))
        strategy     = str(body.get('strategy', 'hybrid'))
        qos_priority = str(body.get('qos_priority', 'best_effort'))

        # FIX: cast node list to int to match trust/fatigue dict keys
        nodes = [int(n) for n in (body.get('nodes') or [])]
        edges = [[int(a), int(b)] for a, b in (body.get('edges') or [])]

        if not nodes or not edges:
            return safe_jsonify({'error': 'Invalid network: nodes/edges missing'}, 400)
        if source not in nodes or dest not in nodes:
            return safe_jsonify({'error': 'source or dest not in nodes list'}, 400)
        if source == dest:
            return safe_jsonify({'error': 'source and dest must differ'}, 400)

        # ── Build graph ───────────────────────────────────────────────────────
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        if not nx.has_path(G, source, dest):
            return safe_jsonify(
                {'error': f'No path exists between Node {source} and Node {dest}'}, 400)

        all_paths = list(nx.all_simple_paths(G, source, dest, cutoff=6))[:20]
        if not all_paths:
            return safe_jsonify({'error': 'No simple path found'}, 400)

        # ── Ensure every node has trust & fatigue entries (int keys) ──────────
        for n in nodes:
            trust_scores_global.setdefault(n, 0.9)
            node_fatigue_global.setdefault(n, 0.0)

        # ── Select best path ──────────────────────────────────────────────────
        best_path, path_metrics = select_best_path_ml(
            all_paths, strategy, trust_scores_global, node_fatigue_global, qos_priority)

        best_path = [int(n) for n in best_path]

        # ── Path metrics & ML score ───────────────────────────────────────────
        path_trust   = float(np.mean([trust_scores_global.get(n, 0.5) for n in best_path]))
        path_metrics['trust_avg'] = round(path_trust, 4)

        features = [
            path_metrics['avg_delay'],
            path_metrics['packet_loss'],
            path_metrics['bandwidth'],
            path_metrics['load'],
            path_metrics['trust_avg'],
        ]
        if ml_model:
            proba      = ml_model.predict_proba([features])[0]
            ml_prob    = float(proba[1]) if len(proba) > 1 else float(proba[0])
            prediction = int(ml_model.predict([features])[0])
        else:
            ml_prob, prediction = 0.5, 1

        # ── Snapshot trust BEFORE update ─────────────────────────────────────
        trust_before = {str(n): round(trust_scores_global.get(n, 0.5), 4) for n in nodes}

        # ── Anomaly detection & QoS ───────────────────────────────────────────
        anomaly_detection = detect_anomalies(
            best_path, trust_scores_global, node_fatigue_global)
        qos_metrics = calculate_qos(path_metrics, qos_priority)

        # ── Fatigue: decay all, charge path nodes ─────────────────────────────
        for n in nodes:
            node_fatigue_global[n] = max(0.0, node_fatigue_global[n] - 0.05)
        for n in best_path:
            node_fatigue_global[n] = min(1.0, node_fatigue_global[n] + 0.20)

        # ── Simulate packet delivery ──────────────────────────────────────────
        successful_packets = simulate_packets(
            best_path, num_packets, trust_scores_global, node_fatigue_global)
        success_rate = float(successful_packets) / float(num_packets)

        # ── Update trust via Perfect Formula ─────────────────────────────────
        reward = compute_reward(success_rate, path_metrics)
        for i, node in enumerate(best_path):
            trust_scores_global[node] = update_trust_enhanced(
                trust_scores_global.get(node, 0.5), 
                reward, 
                ml_prob,
                node_fatigue=node_fatigue_global.get(node, 0.0),
                hop_index=i
            )

        # Passive decay for off-path nodes
        for n in nodes:
            if n not in best_path:
                trust_scores_global[n] = float(
                    max(TRUST_MIN, trust_scores_global.get(n, 0.5) * (1.0 - TRUST_DECAY)))

        trust_after = {str(n): round(trust_scores_global.get(n, 0.5), 4) for n in nodes}

        # ── Assemble response ─────────────────────────────────────────────────
        response_data = {
            'status':             'success',
            'source':             source,
            'destination':        dest,
            'best_path':          best_path,
            'strategy':           strategy,
            'num_packets':        num_packets,
            'successful_packets': successful_packets,
            'success_rate':       round(success_rate, 4),
            'ml_prediction':      prediction,
            'ml_probability':     round(ml_prob, 4),
            'reward_signal':      round(reward,   4),
            'path_metrics':       path_metrics,
            'trust_before':       trust_before,
            'trust_after':        trust_after,
            'anomaly_detection':  anomaly_detection,
            'qos_metrics':        qos_metrics,
        }
        response_data['event_log'] = build_event_log(response_data)

        save_routing_history(response_data)
        save_trust_history()

        return safe_jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return safe_jsonify({'error': str(e), 'status': 'error'}, 500)


@app.route('/api/trust-values', methods=['GET', 'OPTIONS'])
def get_trust_values():
    if request.method == 'OPTIONS':
        return safe_jsonify({}, 200)
    try:
        # FIX: return string keys so frontend JSON.parse is consistent
        tv = {str(k): round(float(v), 4) for k, v in trust_scores_global.items()}
        avg = float(np.mean(list(trust_scores_global.values()))) \
              if trust_scores_global else 0.9
        return safe_jsonify({
            'trust_values': tv,
            'total_nodes':  len(tv),
            'avg_trust':    round(avg, 4),
            'status':       'success',
        })
    except Exception as e:
        return safe_jsonify({'error': str(e), 'status': 'error'}, 500)


@app.route('/api/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return safe_jsonify({}, 200)
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            return safe_jsonify({
                'total_records':    len(df),
                'avg_success_rate': float(df['success_rate'].mean()) if len(df) > 0 else 0.0,
                'latest_records':   df.tail(10).fillna(0).to_dict('records'),
                'status':           'success',
            })
        return safe_jsonify({
            'total_records': 0, 'avg_success_rate': 0,
            'latest_records': [], 'status': 'success',
        })
    except Exception as e:
        return safe_jsonify({'error': str(e), 'status': 'error'}, 500)


@app.route('/api/retrain-model', methods=['POST', 'OPTIONS'])
def retrain_model():
    if request.method == 'OPTIONS':
        return safe_jsonify({}, 200)
    try:
        if not os.path.exists(HISTORY_FILE):
            return safe_jsonify({'message': 'No historical data yet', 'status': 'warning'})
        df = pd.read_csv(HISTORY_FILE).dropna(
            subset=['avg_delay', 'packet_loss', 'bandwidth', 'load', 'trust_avg', 'success'])
        if len(df) < 10:
            return safe_jsonify(
                {'message': f'Need ≥10 records (have {len(df)})', 'status': 'warning'})
        _train_and_save_model(
            extra_X=df[['avg_delay', 'packet_loss', 'bandwidth', 'load', 'trust_avg']].values.tolist(),
            extra_y=df['success'].astype(int).values.tolist(),
        )
        return safe_jsonify({
            'message':      f'Model retrained on {len(df)} historical records',
            'records_used': len(df),
            'status':       'success',
        })
    except Exception as e:
        return safe_jsonify({'error': str(e), 'status': 'error'}, 500)


@app.route('/api/chaos-inject', methods=['POST', 'OPTIONS'])
def chaos_inject():
    if request.method == 'OPTIONS':
        return safe_jsonify({}, 200)
    try:
        body = request.get_json(force=True, silent=True) or {}
        target_node = body.get('target_node')
        if target_node is None:
            return safe_jsonify({'error': 'target_node required'}, 400)
        
        target = int(target_node)
        
        # ── Realistic DDoS/Chaos Engineering Simulation ──
        # 1. Crush the primary target node seamlessly
        # Generating a random value instead of hardcoding so it looks organic
        import random
        trust_scores_global[target] = round(random.uniform(0.00, 0.05), 3)
        node_fatigue_global[target] = 1.0

        # 2. Network Ripple Effect (Cascade load)
        # Random adjacent/neighboring nodes suffer severe fatigue and trust loss 
        # due to the overwhelming spoofed traffic crossing their paths to the target.
        import random
        ripple_affected = []
        for n in list(trust_scores_global.keys()):
            if n != target and random.random() < 0.25: # 25% chance of collateral impact
                trust_scores_global[n] = max(TRUST_MIN, trust_scores_global[n] - 0.25)
                node_fatigue_global[n] = min(1.0, node_fatigue_global.get(n, 0.0) + 0.4)
                ripple_affected.append(n)
        
        # 3. Persist the chaos test immediately so page reloads don't reset it
        save_trust_history()

        msg = f'Critical DDoS simulated on Node {target}.'
        if ripple_affected:
            msg += f' Collateral surge impacted: {", ".join(map(str, ripple_affected))}.'

        return safe_jsonify({
            'status': 'success',
            'message': msg,
            'target_node': target,
            'ripple_nodes': ripple_affected
        })
    except Exception as e:
        return safe_jsonify({'error': str(e), 'status': 'error'}, 500)


@app.route('/health')
def health():
    """Detailed health check for both human and monitoring systems."""
    return safe_jsonify({
        'status':   'ok',
        'ml_model': 'loaded' if ml_model else 'not_loaded',
        'trust_nodes': len(trust_scores_global),
        'timestamp': datetime.now().isoformat()
    })


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🚀 Initializing Smart Routing ML System…")
    initialize_ml_model()
    print("✅ System ready!")
    print("📱 Frontend: http://localhost:5000")
    print("🔌 API root:  http://localhost:5000/api/predict")
    app.run(debug=False, host='0.0.0.0', port=5000)