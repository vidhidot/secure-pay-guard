# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Title
st.title('Secure Pay Guard: Credit Card Fraud Detection')

# Read Data into a Dataframe
#using the smaller file for sample to upload easily on github
df = pd.read_csv('creditcard_sample.csv')


# --- 1 CHECKBOX ---
if st.sidebar.checkbox('Show the initial data set'):
    st.header("Understanding dataset")
    st.write('Initial data set: \n', df)
    st.write('Data decription: \n', df.describe())
    st.write('Shape of the dataframe: ', df.shape)
    st.write('Missing values: ', df.isnull().values.sum())
    st.write('Duplicate rows: ', df.duplicated(keep=False).sum())
    df = df.drop_duplicates() 
    st.write('New data set after removing duplicates:', df)

# --- 2 CHECKBOX ---
if st.sidebar.checkbox('Show the analysis'):
    fraud = df[df.Class == 1]
    valid = df[df.Class == 0]
    outlier_percentage = (df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

    st.header('Univariate analysis')
    st.write('Fraud Cases: ', len(fraud))
    st.write('Valid Cases: ', len(valid))
    st.write('Fraudulent transactions are: %.3f%%' % outlier_percentage)

    def countplot_data(data, feature):
        plt.figure(figsize=(5, 5))
        sns.countplot(x=feature, data=data)
        st.pyplot(plt.gcf())

    st.subheader('Transaction ratio:')
    countplot_data(df, df.Class)

    def graph2():
        f, axes = plt.subplots(ncols=2, figsize=(12, 5))
        colors = ['#C35617', '#FFDEAD']
        sns.boxplot(x="Class", y="Amount", data=df, palette=colors, ax=axes[0], showfliers=True)
        axes[0].set_title('With outliers')
        sns.boxplot(x="Class", y="Amount", data=df, palette=colors, ax=axes[1], showfliers=False)
        axes[1].set_title('Without outliers')
        st.pyplot(f)

    st.subheader('Boxplots for Class vs Amount')
    graph2()

# --- 3 CHECKBOX ---
if st.sidebar.checkbox('Model building on imbalanced data'):
    st.header('Train and test split')
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])

    cols = X_train.columns
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
    X_train[cols] = pt.fit_transform(X_train)
    X_test[cols] = pt.transform(X_test)

# --- 4 CHECKBOX ---
if st.sidebar.checkbox('Compare algorithms'):
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])
    cols = X_train.columns
    pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
    X_train[cols] = pt.fit_transform(X_train)
    X_test[cols] = pt.transform(X_test)

    def visualize_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Oranges')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt.gcf())

    def ROC_AUC(Y, Y_prob):
        fpr, tpr, _ = roc_curve(Y, Y_prob)
        model_auc = roc_auc_score(Y, Y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, label='AUC=%.3f' % model_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot(plt.gcf())

    # Logistic Regression
    st.header('Logistic Regression')
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    st.text(classification_report(y_test, pred))
    visualize_confusion_matrix(y_test, pred)
    ROC_AUC(y_test, proba)

    # Decision Tree (chosen as second model)
    st.header('Decision Tree')
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    st.text(classification_report(y_test, pred))
    visualize_confusion_matrix(y_test, pred)
    ROC_AUC(y_test, proba)

# --- 5 CHECKBOX ---
if st.sidebar.checkbox('Manual transaction verification'):
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    st.title("Manual transaction verification")
    st.write("Input the 30 feature values (excluding Class) separated by commas:")
    input_df = st.text_input('Input All features')
    input_df_lst = input_df.split(',')
    submit = st.button("Submit")
    if submit:
        features = np.array(input_df_lst, dtype=np.float64)
        prediction = model.predict(features.reshape(1, -1))
        if prediction[0] == 0:
            st.success("Legitimate transaction")
        else:
            st.error("Fraudulent transaction")