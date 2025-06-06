# 🏦 Loan Eligibility Predictor

A machine learning project that predicts whether a loan applicant is eligible for a loan, based on financial and personal information. Built using Python, trained with two models (Logistic Regression and Random Forest), and integrated into a web application using Streamlit.

---

## 📌 Problem Statement

Banks and financial institutions need a reliable way to assess whether a loan applicant should be approved or not. This project helps automate that decision-making process using a classification model trained on historical data.

---

## 🚀 Project Highlights

- ✅ Built and compared two models: Logistic Regression and Random Forest
- 📊 Performed thorough Exploratory Data Analysis (EDA) and preprocessing
- 📦 Exported the final model using `joblib`
- 🌐 Integrated the model into a simple web app using Streamlit
- 🔄 Learning to apply MLOps practices step-by-step (version control, model reuse, deployment)

---

## 🛠️ Tech Stack

| Category         | Tools Used                            |
|------------------|----------------------------------------|
| Programming      | Python                                 |
| Data Handling    | Pandas, NumPy                          |
| Visualization    | Matplotlib, Seaborn                    |
| ML Algorithms    | Logistic Regression, Random Forest     |
| Model Saving     | Joblib                                 |
| Deployment       | Streamlit (local, deploy-ready)        |
| Version Control  | Git, GitHub                            |

---

## 🧪 ML Pipeline

1. **Data Preprocessing**
   - Missing values, label encoding, feature selection

2. **EDA**
   - Analysis based on credit history, income, loan amount, etc.

3. **Modeling**
   - Compared Logistic Regression and Random Forest
   - Evaluated with accuracy, precision, recall, and confusion matrix

4. **Model Saving**
   - Saved best-performing model (`loan_model.pkl`) using `joblib`

5. **Web App Integration**
   - Created `app.py` with user input form using Streamlit
   - Model predicts eligibility instantly

---

## 📂 Project Structure

Loan_Eligibility_Predictor/
│
├── app.py # Streamlit web app
├── loan_model.pkl # Saved model
├── loan_prediction.ipynb # Model training & comparison
├── Loan_prediction_eda_part.ipynb # Data analysis
├── loan_prediction_data_understanding.py.ipynb
├── README.md # Project documentation
└── .gitignore


---

## 📈 Results & Insights

- **Random Forest** performed better than Logistic Regression (mention your accuracy/F1-score if available)
- **Credit History**, **Income**, and **Loan Amount** were key features affecting approval
- Visualizations showed a clear relationship between loan status and applicant financial profiles

---

## 💻 How to Run the Project Locally

1. **Clone the repo**

```bash
git clone https://github.com/Shakshi123pal/Loan_Eligibility_Predictor.git
cd Loan_Eligibility_Predictor


2. Install dependencies:
pip install -r requirements.txt


3.Run the Streamlit app:
streamlit run app.py




## 📈 Results & Insights

### Model Performance Summary

| Model               | Accuracy | Precision (Class 0) | Recall (Class 0) | F1-score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
|---------------------|----------|---------------------|------------------|--------------------|---------------------|------------------|--------------------|
| **Random Forest**    | 82.9%    | 0.77                | 0.63             | 0.70               | 0.85                | 0.92             | 0.88               |
| **Logistic Regression** | 85.4%    | 0.95                | 0.55             | 0.70               | 0.83                | 0.99             | 0.90               |

### Detailed Insights

- The **Logistic Regression** model achieved slightly higher accuracy (85.4%) compared to Random Forest (82.9%).
- Logistic Regression has a very high recall (0.99) for the positive class (eligible applicants), meaning it correctly identifies most eligible loan applicants.
- Random Forest shows a better balance between precision and recall for both classes.
- Confusion matrices show:
  - Logistic Regression had fewer false negatives (missed eligible applicants) but more false positives.
  - Random Forest had more balanced errors across both classes.

### Interpretation

These results indicate that Logistic Regression is more sensitive in identifying eligible applicants (good for minimizing missed opportunities), while Random Forest offers a more balanced prediction performance.


