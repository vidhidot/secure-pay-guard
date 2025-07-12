#  Secure Pay Guard

Secure Pay Guard is a machine learning-powered Streamlit web application that detects fraudulent bank card transactions in real-time using behavioral data analysis. This project demonstrates end-to-end data science implementation — from data preprocessing and exploratory analysis to model training, evaluation, and deployment.

##  Features

- Load and analyze credit card transaction data
- Univariate and boxplot visualizations for fraud patterns
- Data preprocessing and transformation pipelines
- Logistic Regression & Decision Tree model comparison
- Manual transaction verification via user input
- Deployed using Streamlit Cloud

##  Dataset

- The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.
- It contains anonymized features (V1 to V28), `Amount`, and a `Class` label indicating fraud (1) or legitimate (0).

##  Tech Stack

- **Frontend**: Streamlit
- **ML Models**: Logistic Regression, Decision Tree
- **Preprocessing**: StandardScaler, PowerTransformer
- **Languages & Libraries**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib





> *Note*: The original CSV file exceeds GitHub’s file size limit. A sampled dataset (`creditcard_sample.csv`) is used for deployment.

