# Diabetes Predictor — Streamlit App

**Diabetes Predictor** is a lightweight Streamlit web app that uses a pre-trained machine learning model to predict whether a person is likely to have diabetes based on clinical features. The app is designed for quick local testing and easy deployment (e.g., on Render.com).

---

## 🚀 Features
- Simple, intuitive Streamlit UI (`frontend.py`) for user input
- Model inference with a pre-trained model file (`diabetes_prediction_model.pkl` or `model.pkl`)
- Clear instructions for local run and Render deployment
- Small helper module (`backend.py`) separates model/loading logic from the UI

---

## 📁 Repo structure
```
.
├─ frontend.py                  # Streamlit app (entrypoint)
├─ backend.py                   # model loading + prediction helpers
├─ diabetes_prediction_model.pkl # trained model (or model.pkl)
├─ requirements.txt
├─ train_test_split.ipynb       # optional: training notebook
└─ README.md
```

---

## ✅ Quick start — run locally

1. Create & activate a virtual environment (recommended)
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run frontend.py
```

4. Open the URL shown by Streamlit (usually `http://localhost:8501`)

---

## 🛠 Deployment on Render (Streamlit web service)

1. On Render, choose **New → Web Service** (do **NOT** choose Static Site).
2. Connect your GitHub repo.
3. Set fields:
   - **Environment:** Python 3
   - **Build Command:**  
     ```
     pip install -r requirements.txt
     ```
   - **Start Command:**  
     ```
     streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
     ```
4. Create the service — Render will build and expose the app at a public URL.

> No publish directory is needed because Streamlit runs as a web service.

---

## 🔎 Important notes

- **Model filename**: Confirm the filename your code expects (e.g., `diabetes_prediction_model.pkl` or `model.pkl`) and that same file is in the repo root.
- **Relative path safety**: Use `os.path.join(os.path.dirname(__file__), 'model.pkl')` in `backend.py` to reliably load the model.
- **Large model**: If the `.pkl` is very large (>50–100 MB), consider hosting it externally (S3) and downloading during the first run, or compressing with `joblib` + compression.

---

## 🧪 Troubleshooting

**Missing package errors on Render**
- Ensure every library you import in `frontend.py` and `backend.py` is listed in `requirements.txt`.

**Model not found**
- Make sure `backend.py` loads the model using a path relative to the repo root (see snippet below).

**Port or binding errors**
- Always include `--server.port $PORT --server.address 0.0.0.0` in the start command for Render.


---

## 📦 Requirements.txt
```
streamlit
pandas
numpy
scikit-learn
joblib
```
---

## 📝 License & author
- Author: _PARV JAIN_
- License: MIT 
