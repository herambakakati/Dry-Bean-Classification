 🌱 Dry Bean Classification using Machine Learning

 📌 Project Overview

This project focuses on classifying different types of dry beans using Machine Learning techniques. The model uses geometric and shape-based features of beans to accurately predict their class.

The project covers:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model building and evaluation
* Handling class imbalance (SMOTE)
* Deployment using Streamlit

---

 📊 Dataset Details

* Total Records: **13,611**
* Features: **16 numerical features + 1 target**
* Classes: **7 bean types**

 Target Classes:

* DERMASON
* SIRA
* SEKER
* HOROZ
* CALI
* BARBUNYA
* BOMBAY

📎 Reference: 

---

 🔍 Key Features Used

* Area
* Perimeter
* MajorAxisLength
* MinorAxisLength
* AspectRatio
* Eccentricity
* ConvexArea
* EquivDiameter
* Extent
* Solidity
* Shape Factors

---

 🧠 Project Workflow

 1. Data Preprocessing

* Removed duplicates
* Checked missing values (none found)
* Feature scaling using StandardScaler
* Label Encoding for target variable

 2. Exploratory Data Analysis (EDA)

* Distribution plots & boxplots
* Skewness analysis
* Correlation heatmap
* Pairplot visualization

 Insights:

* No missing data
* Moderate class imbalance
* High correlation among size-related features
* Outliers present but represent real-world variations

---

 ⚙️ Feature Engineering

* Removed highly correlated features (>0.95)
* Reduced dimensionality from 17 → 10 features

---

 ⚖️ Handling Class Imbalance

* Used:

  * SMOTE (Synthetic Minority Oversampling)
  * Random Oversampling
  * Random Undersampling

---

 🤖 Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SGD)
* Naive Bayes

---

 📈 Model Performance Summary

 Without SMOTE:

* Random Forest: **91.84%**
* Gradient Boosting: **91.73%**
* Logistic Regression: **91.44%**

 With SMOTE:

* Random Forest: **91.73%**
* Gradient Boosting: **91.66%**

📌 Insight:

* Models performed well even without SMOTE
* SMOTE improved minority class handling

---

 🏆 Best Model

**Random Forest Classifier**

* High accuracy
* Robust to outliers
* Stable performance

---

 💻 Streamlit App Features

* User input-based prediction
* Interactive UI
* Real-time classification
* Visual analytics

---

 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Dry-Bean-Classification.git
cd Dry-Bean-Classification
```

 2. Install Requirements

```bash
pip install -r requirements.txt
```

 3. Run Streamlit App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── data/
├── notebooks/
├── models/
├── app.py
├── requirements.txt
└── README.md
```

---

 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn
* Streamlit

---

 📌 Key Learnings

* Handling imbalanced datasets using SMOTE
* Importance of feature correlation analysis
* Model comparison and evaluation using F1-score
* End-to-end ML pipeline building
