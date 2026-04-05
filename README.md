#  Smart Routing ML System

A machine learning–based network routing simulator that selects optimal paths using **trust scoring, QoS metrics, and anomaly detection**.

---

## Features

* **Intelligent Routing:** Hybrid ML + cost-based path selection
* **Adaptive Trust System:** Nodes update trust based on performance
* **Packet Simulation:** Models delay, loss, congestion, and fatigue
* **Anomaly Detection:** Flags unreliable or overused nodes
* **QoS Evaluation:** Supports real-time, premium, and best-effort traffic
* **Persistent Learning:** Stores history and retrains ML model

---

## 🏗️ Project Structure

```
CN_PROJECT/
│
├── data/               # Routing history & trust values
├── models/             # Trained ML models
├── src/
│   ├── templates/
│   │   └── index.html  # Frontend UI
│   ├── app.py          # Flask backend
│   ├── main.py         # Core simulation logic
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/smart-routing-ml.git
cd smart-routing-ml

python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Run

```bash
cd src
python app.py
```

Open: **http://localhost:5000**

---

## 🔌 API (Main)

### POST `/api/predict`

```json
{
  "source": 0,
  "dest": 5,
  "nodes": [0,1,2,3,4,5],
  "edges": [[0,1],[1,2],[2,5]],
  "num_packets": 10
}
```

---

## 🧪 Tech Stack

* Python, Flask
* Scikit-learn (Random Forest)
* NetworkX
* Pandas, NumPy
* HTML, CSS, JS

---

## ⚠️ Notes

* Designed for simulation and learning
* ML accuracy improves with more historical data

---

## 👨‍💻 Author

Adarsh – B.Tech Student | AI & Systems Enthusiast

---

A system-level project combining networking + ML + behavioral logic
