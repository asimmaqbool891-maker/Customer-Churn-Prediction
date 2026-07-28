# 📡 Customer Churn Prediction System

A production-ready, portfolio-quality **Streamlit** dashboard that predicts
whether a telecom customer is likely to **churn**, powered by a fully-fitted
**scikit-learn Pipeline** (preprocessing + Logistic Regression).

![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)

---

## 📋 Project Description

Customer churn is one of the most critical metrics for subscription-based
telecom businesses. This application allows a user (e.g. a customer support
agent or business analyst) to enter a customer's profile — demographics,
subscribed services, contract details, and billing information — and
instantly receive:

- A **churn / stay prediction**
- The underlying **probability scores**
- A **risk classification** (Low / Medium / High)
- Rich **visual analytics** (gauge chart, probability donut, risk indicator)
- A **downloadable CSV report** of the prediction

The trained model is a scikit-learn `Pipeline` that bundles:

1. **ColumnTransformer**
   - Numerical features (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) → `PowerTransformer` + `StandardScaler`
   - Ordinal features (`PaymentMethod`, `Contract`, `InternetService`, `MultipleLines`) → `OrdinalEncoder`
   - Categorical features (`gender`, `Partner`, `Dependents`, `PhoneService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`) → `OneHotEncoder`
2. **LogisticRegression** classifier

The Streamlit app **never** performs manual encoding or scaling — it only
constructs a raw `DataFrame` with the original feature names and passes it
directly to `model.predict()` / `model.predict_proba()`, letting the
pipeline handle 100% of the preprocessing.

---

## ✨ Features

- 🎨 Premium **dark-themed**, glassmorphism UI with gradient header
- 🧭 Fully organized **sidebar** input form (selectboxes, sliders, number inputs)
- 🔮 One-click **churn prediction**
- 📊 Interactive **Plotly** visualizations: gauge chart, probability donut, risk bar
- 🚦 Color-coded **risk levels** (Green / Yellow / Red)
- 🧾 **Customer summary** table of all entered data
- ⬇️ **CSV export** of prediction results
- 🛡️ Robust **exception handling** (missing model file, prediction errors, unseen categories)
- 🧹 Clean, PEP8-compliant, fully commented, modular Python code

---

## 🗂️ Project Structure

```
customer-churn-prediction/
├── app.py                  # Main Streamlit application
├── style.css                # Custom dark glassmorphism theme
├── requirements.txt         # Python dependencies
├── README.md                 # Project documentation
└── Telco_Churn_LR(1).pkl    # Trained sklearn Pipeline (add this file yourself)
```

---

## ⚙️ Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- plotly

All dependencies are listed in [`requirements.txt`](./requirements.txt).

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add the trained model**

   Place your trained pipeline file `Telco_Churn_LR(1).pkl` in the project
   root folder (same directory as `app.py`).

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The app will open automatically in your default browser at
`http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this project (including `Telco_Churn_LR(1).pkl`) to a **public or
   private GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select your repository, branch, and set the main
   file path to `app.py`.
4. Click **"Deploy"** — Streamlit Cloud will automatically install
   dependencies from `requirements.txt` and launch the app.

> **Note:** If your model file exceeds GitHub's file size limits, use
> [Git LFS](https://git-lfs.com/) or host the `.pkl` file externally and
> download it at app startup.

---

## 🖼️ Screenshots

> _Add your screenshots below after running the app locally._

| Dashboard Home | Prediction Result |
|---|---|
| `screenshots/home.png` | `screenshots/result.png` |

| Gauge & Risk Charts | Customer Summary |
|---|---|
| `screenshots/charts.png` | `screenshots/summary.png` |

---

## 🧠 Model Input Schema

The pipeline expects a DataFrame with **exactly** these 19 columns, in any
order (the app enforces the correct order internally):

| Column | Type | Example |
|---|---|---|
| gender | categorical | `Female`, `Male` |
| SeniorCitizen | binary (0/1) | `0` |
| Partner | categorical | `Yes`, `No` |
| Dependents | categorical | `Yes`, `No` |
| tenure | numeric | `12` |
| PhoneService | categorical | `Yes`, `No` |
| MultipleLines | categorical | `No`, `Yes`, `No phone service` |
| InternetService | categorical | `DSL`, `Fiber optic`, `No` |
| OnlineSecurity | categorical | `Yes`, `No`, `No internet service` |
| OnlineBackup | categorical | `Yes`, `No`, `No internet service` |
| DeviceProtection | categorical | `Yes`, `No`, `No internet service` |
| TechSupport | categorical | `Yes`, `No`, `No internet service` |
| StreamingTV | categorical | `Yes`, `No`, `No internet service` |
| StreamingMovies | categorical | `Yes`, `No`, `No internet service` |
| Contract | categorical | `Month-to-month`, `One year`, `Two year` |
| PaperlessBilling | categorical | `Yes`, `No` |
| PaymentMethod | categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| MonthlyCharges | numeric | `70.35` |
| TotalCharges | numeric | `845.5` |

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit + custom CSS (glassmorphism, gradients)
- **Visualization:** Plotly
- **Machine Learning:** scikit-learn (Pipeline + Logistic Regression)
- **Serialization:** joblib

---

## 📄 License

This project is provided as-is for educational and portfolio purposes.
Feel free to fork, modify, and use it in your own projects.

---

## 🙋 Author

Built with ❤️ as a demonstration of production-grade ML application
engineering — combining a clean scikit-learn pipeline with a polished,
professional Streamlit front end.