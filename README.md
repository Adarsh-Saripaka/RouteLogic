# 🚀 Smart Routing ML System

A machine learning–based network routing simulator that selects optimal paths using **trust scoring, QoS metrics, anomaly detection, and dynamic traffic simulation**.

---

## 📌 Features

* **Intelligent Routing:** Hybrid ML + cost-based path selection
* **Adaptive Trust System:** Nodes update trust based on performance
* **Packet Simulation:** Models delay, loss, congestion, and fatigue
* **Anomaly Detection:** Flags unreliable or overused nodes
* **QoS Evaluation:** Supports real-time, premium, and best-effort traffic
* **Traffic / DoS Simulation:** Inject artificial load on specific nodes to simulate congestion or denial-of-service scenarios and observe routing behavior
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
│   ├── requirements.txt
│   └── vercel.json
│
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Adarsh-Saripaka/RouteLogic.git

pip install -r src/requirements.txt
```

---

## ▶️ Run (Local)

```bash
cd src
python app.py
```

Open: **http://localhost:5000**

---

## 🌐 Live Deployment

https://route-logic.vercel.app/

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
* HTML, CSS, JavaScript

---

## ⚠️ Notes

* Designed for simulation and learning
* ML accuracy improves with more historical data
* Data persistence (trust/history) is limited in deployment due to serverless environment

---

## 👨‍💻 Authors

* Adarsh Saripaka
* Yug Patel

---

## 📚 References

1. **Kurose, J. F., & Ross, K. W.** (2020). *Computer Networking: A Top-Down Approach* (8th ed.). Pearson. *(For core networking, QoS analysis, and routing concepts)*
2. **Boutaba, R., et al.** (2018). *A comprehensive survey on machine learning for networking: evolution, applications and research opportunities.* Journal of Internet Services and Applications, 9, 1-99. *(For ML integration in network infrastructure)*
3. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. *(For foundational concepts on Decision Trees and heuristic algorithms)*
4. **Dijkstra, E. W.** (1959). *A note on two problems in connexion with graphs.* Numerische Mathematik, 1(1), 269–271. *(For the foundational shortest-path routing algorithm)*
5. **Hagberg, A., Swart, P., & S Chult, D.** (2008). *Exploring network structure, dynamics, and function using NetworkX.* *(Official reference for the NetworkX framework used in the simulation)*

---
