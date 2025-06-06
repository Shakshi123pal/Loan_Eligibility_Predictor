

# # DATA CLEANING AND PREPROCESSING
# There start **Data cleaning & Preprocessing** part  after **Data understanding** and **Eda** 
# -  🔍 [Data Understanding](Loan_prediction_Data_understanding.ipynb)
# -  📊[Exploratory Data Analysis (EDA)](Loan_prediction_eda_part.ipynb)
# 
# ## Goals:
# basically in this notebook,we clean and prepare the data before applying **Machine learning models**
# - Handle missing Values
# - correct data types
# - Handle outliers if needed
# - Encode Categorical variables
# - Normalize or scale numerical values(if needed)




## import libraries and Data


# Load dataset
df=pd.read_csv('train_loandataset.csv')
df.head()





columns=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status']
for col in columns:
    print(df[col].value_counts())





# 3+ replace by 3
df['Dependents'].replace('3+', 3, inplace=True)





df['Dependents'].value_counts()





# handle missing value
df.isnull().sum()





# find missing values in some column
# Gender-13
# Married-3
# Dependents-15
# Self_Employed-32
# LoanAmount-22
# Loan_Amount_Term-14
# creadit_History-50


# ### 🔍 Missing Values Handling Summary:
# 
# We handled missing values using the following strategies:
# - Mode for categorical columns (like Gender, Married)
# - Median for numerical columns with skew (LoanAmount)
# - Mode for critical features like `Credit_History` to avoid introducing bias
# 
# This step ensures that the dataset is ready for encoding and modeling without losing rows unnecessarily.




# so there i fill all missing value
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)





df.isnull().sum()




# now check dtype and then handle if needed
df.dtypes


# ###  Data Type Correction
# To optimize performance and ensure correct interpretation of features:
# - Categorical features like `Gender`, `Married`, `Education`, `Self_Employed`, and `Property_Area` are converted to `"category"` type.
# - Numerical float features that are logically integers (`Dependents`,`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`) are cast to `"int"`.
# - Left `Loan_ID` unchanged as it's an identifier and won't be used for modeling




# Convert to 'category' type
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Convert numerical float columns to integer
int_columns = ['Dependents','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in int_columns:
    df[col] = df[col].astype(int)

# Check dtypes again
df.dtypes





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace 'df' with your actual DataFrame name
numeric_cols = df.select_dtypes(include=['int32']).columns

# Set the plot style
sns.set(style="whitegrid")

# Loop through all numeric columns and plot boxplots
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()





#  check outliers
outlier_info = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_info[col] = len(outliers)

# Capping outliers at 1st and 99th percentiles
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    lower_limit = df[col].quantile(0.01)
    upper_limit = df[col].quantile(0.99)
    df[col] = df[col].clip(lower_limit, upper_limit)





import numpy as np

# Apply log1p (log(1 + x)) to avoid log(0) issues
df['ApplicantIncome_log'] = np.log1p(df['ApplicantIncome'])
df['CoapplicantIncome_log'] = np.log1p(df['CoapplicantIncome'])

df['LoanAmount_log'] = np.log1p(df['LoanAmount'])




df.head()





print(df.shape)  # Should be (614, X) where X = total columns after transformation





# label encoding for binary Columns
label_map = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Loan_Status': {'Y': 1, 'N': 0}  # target variable
}

for col, mapping in label_map.items():
    df[col] = df[col].map(mapping)





df.head()





# One-Hot Encoding for Multi-Class Columns
df = pd.get_dummies(df, columns=['Property_Area', 'Dependents'], drop_first=True)





df.head()


#  Feature Engineering part start




# Models may perform better when both incomes are considered together.
df['Total_Income']=df['ApplicantIncome']+df['CoapplicantIncome']
df['Total_Income_log'] = np.log1p(df['Total_Income'])




df.head()




df.head()





df.info()





# EMI helps estimate affordability. Higher EMI compared to income may reduce eligibility.
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']





# Tells how much income is left after paying EMI — useful for judging risk.
df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 1000)  # if LoanAmount in thousands




df.head()

# Now drop raw columns
df.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income'], axis=1, inplace=True)





df.head()





from sklearn.model_selection import train_test_split

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training rows: {X_train.shape[0]}")
print(f"Testing rows: {X_test.shape[0]}")




from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))




print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))





from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)





from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




import matplotlib.pyplot as plt
import numpy as np

metrics = [
    "Accuracy", "Precision (Class 0)", "Recall (Class 0)", "F1-Score (Class 0)",
    "Precision (Class 1)", "Recall (Class 1)", "F1-Score (Class 1)",
    "Macro Avg F1-Score", "Weighted Avg F1"
]

logistic_scores = [0.85, 0.95, 0.55, 0.70, 0.83, 0.99, 0.90, 0.80, 0.84]
rf_scores = [0.84, 0.78, 0.66, 0.71, 0.86, 0.92, 0.89, 0.80, 0.83]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, logistic_scores, width, label='Logistic Regression')
rects2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()

plt.ylim(0, 1)
plt.tight_layout()
plt.show()



best_model=model
import joblib
joblib.dump(model, 'loan_model.pkl')

import os
os.listdir()







