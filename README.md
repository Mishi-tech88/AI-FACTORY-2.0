# 🏭 AI Factory Intelligence System – Complete README & Project Description

## 📌 Project Overview

The **AI Factory Intelligence System** is an end-to-end, industry‑style AI platform designed for a manufacturing company. It ingests **multiple data modalities** (tabular production records, time‑series sensor readings, product images, maintenance text, and PDF manuals) and combines them into a **single, integrated command center**.

The system answers four critical questions for factory operators:

1. **What is happening now?** – Real-time defect detection, failure probability.
2. **What will go wrong next?** – Predictive maintenance and anomaly forecasting.
3. **Why does the AI believe this?** – Explainable AI (SHAP, Grad‑CAM, feature importance).
4. **What action should be taken?** – Multi‑agent decision‑making backed by a digital twin simulation and human‑in‑the‑loop review.

All components are integrated into a **Streamlit web application** with a downloadable PDF report, making it suitable for plant managers, maintenance engineers, and quality control teams.

---

## 🚀 Key Features

- **Data Engineering & EDA** – Automated pipeline that generates realistic synthetic factory data, handles missing values, outliers, feature engineering, and temporal train/val/test splits.
- **Machine Learning & Deep Learning** – Baseline models (Random Forest, XGBoost) and deep learning (ANN for tabular, LSTM/MLP for time‑series, ResNet with transfer learning for images).
- **Computer Vision** – Product defect classification, Grad‑CAM heatmaps for explainability, severity estimation.
- **Natural Language Processing** – Entity extraction from maintenance notes, similarity search, and integration with RAG.
- **Generative AI & RAG** – Retrieval‑augmented generation using machine manuals and SOPs (PDFs) to ground AI answers in factual evidence.
- **Agentic AI / Multi‑Agent System** – Four cooperative agents (Vision, Predictive Maintenance, Knowledge, Planning, Explainability, and Simulation) that pass structured information and produce a final recommendation.
- **Digital Twin Simulation** – A discrete‑time simulation that compares operational scenarios (continue, stop, reduce load) and recommends the most cost‑effective action.
- **Explainable AI** – SHAP for tabular, Grad‑CAM for images, and feature coefficients for time‑series, all summarized in human‑readable explanations.
- **Human‑in‑the‑Loop** – Supervisor can **Approve**, **Reject**, or **Modify** the AI recommendation; every decision is logged with timestamp and feedback.
- **MLOps with MLflow** – Experiment tracking, parameter logging, metric comparison, and model registration (best model saved).
- **Web Application & Reporting** – Interactive Streamlit dashboard to upload images and CSV data, view predictions, explanations, simulation results, review decisions, and generate a PDF report.

---

## 🧱 Architecture

The system is built as a modular Python project with a clear separation of concerns:

```
AI_FACTORY_SYSTEM/
├── data/                    # Raw and processed data (auto-generated)
├── models/                  # Saved models (pickle, pytorch)
├── src/
│   ├── data_generation.py   # Synthetic data generation
│   ├── preprocess.py        # Cleaning, merging, feature engineering
│   ├── train_tabular.py     # Random Forest, XGBoost, ANN (MLflow)
│   ├── train_timeseries.py  # Logistic regression on sequences (MLflow)
│   ├── train_image.py       # ResNet with transfer learning (MLflow)
│   ├── rag_pipeline.py      # RAG knowledge base (FAISS + FLAN‑T5)
│   ├── vision_advanced.py   # Grad‑CAM, severity, autoencoder
│   ├── nlp_pipeline.py      # Entity extraction, similarity
│   ├── digital_twin.py      # Factory simulator
│   ├── agent_system.py      # Multi‑agent orchestrator
│   ├── explainability.py    # SHAP, Lime, Grad‑CAM wrappers
│   ├── human_review.py      # CLI human review
│   └── report_generator.py  # PDF report generation
├── app.py                   # Streamlit web application
├── requirements.txt         # Dependencies
└── README.md                # This file
```

### Agent Communication Flow

The multi‑agent system uses a **shared state dictionary**:

1. **Vision Agent** – analyzes image → defect probability, severity.
2. **Predictive Maintenance Agent** – analyzes sensor sequence → failure probability.
3. **Knowledge Agent** – uses RAG to retrieve relevant SOP/manual excerpts.
4. **Simulation Agent** – runs digital twin for three scenarios → cost, production, risk.
5. **Planning Agent** – combines all inputs → final action, reasoning, confidence.
6. **Explainability Agent** – generates SHAP, Grad‑CAM, and coefficient explanations.
7. **Human Review** – (optional) supervisor approves/rejects/modifies the decision.

All agents are stateless and can be scaled or replaced independently.

---

## 🛠️ Tech Stack

- **Languages**: Python 3.10+
- **Data Processing**: Pandas, NumPy, Scikit‑learn
- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, Pillow, Grad‑CAM
- **NLP & RAG**: Sentence‑Transformers, FAISS, HuggingFace Transformers (FLAN‑T5)
- **Explainability**: SHAP, LIME
- **MLOps**: MLflow
- **Web Framework**: Streamlit
- **Reporting**: FPDF
- **Visualization**: Matplotlib, Seaborn

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Git (optional)

### Steps

1. **Clone the repository** (or copy the project folder):
   ```bash
   git clone https://github.com/yourusername/ai-factory-system.git
   cd ai-factory-system
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model** (for NLP):
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Generate the data and train all models** (optional – you can skip if pre‑trained models are provided):
   ```bash
   python run_all_pipeline.py
   ```
   This runs data generation, preprocessing, and all training scripts with MLflow logging.

---

## ▶️ Usage

### Command‑Line Demo (Quick Test)
```bash
python src/agent_system.py
```
This runs the agent system on a random image from the `data/images/` folder with dummy sensor data and prints the final recommendation and explanations.

### Web Application (Full Interface)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.  
- Upload a product image (`.png`, `.jpg`, `.jpeg`) – required.
- Optionally upload a sensor CSV (with columns: `vibration`, `pressure`, `rpm`). If none, dummy data is used.
- Click **Run AI Analysis**.
- Review the recommendation, confidence, metrics, explanations, and simulation results.
- **Approve**, **Reject**, or **Modify** the recommendation.
- Generate a **PDF report** containing all details.

### MLflow Dashboard
```bash
mlflow ui
```
View experiment tracking, compare model metrics, and register the best model.

---

## 📊 Example Results

After running the system on a product image, you might see:

- **Defect Probability**: 21.57%
- **Failure Probability**: 62.66%
- **Recommendation**: *Reduce load (simulation recommends)* – because simulation shows:
  - `continue`: production 2074, cost 148, risk 0.25
  - `stop`: production 0, cost 411, risk 0.23
  - `reduce_load`: production 6635, cost 30, risk 0.004
- **Explanations**:
  - Image: Grad‑CAM highlights defective regions.
  - Tabular: SHAP shows `vibration_rolling_std_24` and `temperature` as top contributors.
  - Time‑series: coefficients indicate `t0_rpm` and `t18_rpm` are most influential.

---

## 🧪 Experiments with MLflow

Three meaningful experiments were conducted:

| Model | Accuracy | F1 | ROC‑AUC |
|-------|----------|----|---------|
| Random Forest (tabular) | 0.83 | 0.78 | 0.89 |
| XGBoost (tabular)      | 0.85 | 0.82 | 0.92 |
| ANN (tabular)          | 0.80 | 0.76 | 0.87 |

The best model (XGBoost) was registered in MLflow as `FactoryDefectDetector`.

For time‑series:
- Logistic Regression (stable, no NaN) achieved ROC‑AUC 0.89.
- LSTM (tried but unstable) showed that simple models can outperform deep ones on small data.

For images:
- ResNet‑18 with transfer learning achieved >90% accuracy and high F1 on defect detection.

---

## 🤝 Human‑in‑the‑Loop

The system never makes unsupervised decisions. After the AI recommendation is displayed, a human supervisor must **Approve**, **Reject**, or **Modify** it. The decision is logged to `human_decisions.jsonl` with:
- Timestamp
- AI recommendation and reasoning
- Human status
- Modified action (if any)
- Feedback (optional)

---

## 🔍 Explainability

The system provides multi‑faceted explanations:

- **Tabular**: SHAP values (or fallback to coefficients) show feature importance.
- **Image**: Grad‑CAM heatmaps highlight regions that influenced the prediction.
- **Time‑series**: Regression coefficients or SHAP values identify which past time steps were most important.
- **Overall**: A natural‑language summary combines all explanations.

---

## 🏆 Future Work

- **Real‑time data ingestion** from MQTT/OPC‑UA or IoT sensors.
- **Deploy as a microservice** using FastAPI or Docker.
- **Add more simulation parameters** (energy consumption, worker efficiency).
- **Auto‑ML for hyperparameter tuning**.
- **Integration with ERP/MES systems** for seamless factory operations.
- **Enhanced report generation** with interactive charts and dashboards.

---

## 🖼️ Screenshots

<img width="1127" height="677" alt="res-3" src="https://github.com/user-attachments/assets/d44ed596-d020-4526-a88a-db34ccea4f6d" />
<img width="1170" height="706" alt="res-2" src="https://github.com/user-attachments/assets/b79e09a6-d032-4c3d-9d15-45988708ac1d" />
<img width="1190" height="833" alt="res-1" src="https://github.com/user-attachments/assets/fb79a8f5-93d8-4ace-8626-309453dee9ea" />


## 📄 License

This project is open‑source and available under the MIT License. See the [LICENSE](LICENSE) file for details.

---


## 📧 Contact

For questions, suggestions, or collaborations, please reach out at [your.email@example.com](mailto:your.email@example.com).

---

**Built with ❤️ for the future of smart manufacturing.**
