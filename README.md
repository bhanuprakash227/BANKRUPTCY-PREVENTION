# 🏦 Bankruptcy Prevention

Predict whether a company will go **bankrupt** or **not** from six qualitative risk indicators. The project covers exploratory data analysis, feature engineering, model comparison and a deployed **Streamlit** web app.

🔗 **Live app:** https://bankruptcy-prevention-1442.streamlit.app

> Free Streamlit apps go to sleep after a few days without visitors. If you see a "wake up" button, click it and wait about a minute.

---

## 📌 Business objective

This is a binary classification problem: model the probability that a business goes bankrupt (`bankruptcy` vs `non-bankruptcy`) so that high-risk companies can be flagged early.

## 📊 Data set

250 companies, 6 features and 1 target. Every feature takes the value 0, 0.5 or 1.

| Column | Meaning |
|---|---|
| `industrial_risk` | 0 = low risk, 0.5 = medium, 1 = high risk |
| `management_risk` | 0 = low risk, 0.5 = medium, 1 = high risk |
| `operating_risk` | 0 = low risk, 0.5 = medium, 1 = high risk |
| `financial_flexibility` | 0 = low, 0.5 = medium, 1 = high flexibility |
| `credibility` | 0 = low, 0.5 = medium, 1 = high credibility |
| `competitiveness` | 0 = low, 0.5 = medium, 1 = high competitiveness |
| `class` | **target**: `bankruptcy` / `non-bankruptcy` |

For the three *risk* columns higher is worse; for the three *strength* columns (flexibility, credibility, competitiveness) higher is better.

## 🔍 Key EDA findings

- No missing values, invalid values or outliers.
- Classes are fairly balanced: 57% non-bankruptcy, 43% bankruptcy.
- 147 of the 250 rows are duplicates (103 unique), with **no conflicting labels**.
- `competitiveness`, `financial_flexibility` and `credibility` are by far the strongest signals. Every company with `competitiveness = 0` went bankrupt; none with `competitiveness = 1` did.

## 🛠️ Feature engineering

Four features are added to the original six (see `features.py`):

| Feature | Definition |
|---|---|
| `risk_score` | mean of the three risk features |
| `strength_score` | mean of the three strength features |
| `net_score` | `strength_score` − `risk_score` |
| `weak_count` | number of strength features at the lowest level (0 to 3) |

## 🤖 Modelling

- 11 algorithms (Logistic Regression, KNN, SVM, Naive Bayes, LDA, MLP, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting; plus XGBoost when installed), each tried on the **original** and the **engineered** features.
- Evaluation with 5-fold cross-validation repeated 5 times, using **group-aware splits** so that identical rows never appear in both the training and test part (otherwise scores look better than they really are).
- Best model chosen by F1, with ROC-AUC and fold-to-fold stability as tie-breakers, then tuned with `GridSearchCV`.

**Result:** the best model was **Linear Discriminant Analysis (LDA)** with cross-validated F1 ≈ 0.995, accuracy ≈ 99.6% and ROC-AUC ≈ 1.0. Several other models (Logistic Regression, neural network, Extra Trees) are within noise of it, so the data is easy to separate rather than the model being "perfect".

## 📁 Repository structure

```
├── app.py                 # Streamlit web app
├── features.py            # feature engineering functions (shared by training and app)
├── model.joblib           # trained scikit-learn pipeline
├── requirements.txt       # dependencies (scikit-learn version must match the model)
├── Bankruptcy_prevention_Dataset.csv   # data set (optional, for reference)
└── Bankruptcy_Prevention_End_to_End_Colab.ipynb   # EDA + modelling notebook (optional)
```

## 🚀 Run locally

```bash
git clone https://github.com/bhanuprakash227/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 🖥️ App features

- **Single company:** choose Low / Medium / High for each of the six factors and get the bankruptcy probability with a risk label.
- **Many companies (CSV):** upload a CSV containing the six feature columns (`;` or `,` separated) and download predictions for every row.
- **About:** short description of the model and inputs.

## 🔁 Re-train the model

Open the notebook in Google Colab, upload `Bankruptcy_prevention_Dataset.csv`, run all cells, then copy the generated `model.joblib`, `features.py` and `requirements.txt` into this repository.

## ⚠️ Limitations

- Only 250 rows (103 unique feature combinations).
- One feature (`competitiveness`) almost decides the outcome, so the ~99% scores are probably optimistic for companies unlike those in the data.
- Inputs are qualitative judgements, so treat the output as decision support, not a final verdict.

## 🧰 Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib / Seaborn · Streamlit

## 👤 Author

**Ravula Bhanu Prakash**
