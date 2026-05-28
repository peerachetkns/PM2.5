# 🌦️ Utilizing Big Data and Machine Learning for Air Pollution Monitoring: Air Quality Classification

A Senior Project by undergraduate students in the Computer Engineering program at Mae Fah Luang University. This repository implements an end-to-end Machine Learning and Deep Learning pipeline to monitor, analyze, and classify air quality levels based on meteorological conditions and pollutant concentrations.

---

## 📌 Project Objectives
* To analyze the relationship between environmental/meteorological parameters and PM2.5 levels.
* To investigate and compare the performance of various classification algorithms, feature scaling techniques, and data transformations.
* To implement and evaluate state-of-the-art oversampling techniques to handle severe class imbalance within air quality datasets.

## 📊 Dataset & Pipeline
This project utilizes the **Beijing Multi-Site Air Quality Dataset** (spanning March 1, 2013, to February 28, 2017), comprising 420,769 hourly observations and 18 attributes.

### 1. Data Preprocessing
* **Missing Values Handling:** Removed records with missing target values (PM2.5). Replaced missing pollutant variables (PM10, SO2, NO2, CO, O3) with their median. Meteorological factors (TEMP, PRES, DEWP) and rainfall (if missing > 5%) were filled using linear interpolation. Wind direction (wd) and station names were imputed using the mode.
* **Outlier Handling:** Removed negative values for pollutants and capped extreme anomalies based on realistic thresholds (e.g., PM2.5 <= 1000, PM10 <= 1500, CO <= 5000 µg/m³). Meteorological variables were restricted to physically realistic ranges.
* **Feature Selection & Deduplication:** Dropped irrelevant columns (No, year, month, day, hour, station) and removed duplicate rows using multiple key attributes to prevent model bias.

### 2. Feature Engineering
* **Trigonometric Encoding:** Transformed the categorical wind direction (`wd`) into continuous trigonometric components (`wd_sin` and `wd_cos`) to properly preserve its cyclic spatial orientation (0°-360°).
* **Normalization:** Applied Min-Max Scaling to standardize all numerical features into a [0, 1] range, ensuring optimal convergence for gradient-based models.
* **Target Label Encoding (Multi-Class):** Converted continuous PM2.5 concentrations into a discrete **Five-Class Categorization** strictly aligned with the Thai Air Quality Standard issued by the Pollution Control Department (PCD), Thailand:
  * Level 1: Good (<= 15)
  * Level 2: Moderate (15 - 25)
  * Level 3: Unhealthy for Sensitive Groups (25 - 37.5)
  * Level 4: Unhealthy (37.5 - 75)
  * Level 5: Very Unhealthy (> 75)

## ⚖️ Class Imbalance Management
Due to the highly skewed distribution of PM2.5 levels (specifically minority Levels 2 and 3), two oversampling strategies were applied strictly to the training set (80/20 split) using five synthetic data generation frameworks (SMOTE, Borderline-SMOTE, SMOTE-Tomek, ADASYN, and KMeans-SMOTE):
1. **Selective Oversampling:** Artificially expanded only the underrepresented categories (Levels 2 & 3) to 50% of the majority class size.
2. **Full Equalization:** Standardized all class representations uniformly to match the exact size of the majority class (Level 5).

## 🤖 Models Evaluated
Models were rigorously trained and evaluated using 10-Fold Cross-Validation to ensure statistical reliability and prevent overfitting:
* **Classical Machine Learning:** Decision Tree (DT), Random Forest (RF), Extra Trees (ET), Gradient Boosting (GB), Support Vector Machine (SVM), K-Nearest Neighbors (KNN).
* **Deep Learning Frameworks:** Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM).

## 🏆 Key Results & Performance
The comprehensive evaluation revealed that ensemble tree models significantly outperformed others. The best performing model was **Extra Trees (ET) combined with Borderline-SMOTE under the Full Equalization strategy**. 

Validation Performance Metrics for the best model:
* **Accuracy:** 92.7%
* **Precision:** 92.8%
* **Recall:** 92.7%
* **F1-Score:** 92.7%
* **ROC-AUC:** 99.3%

The ensemble tree structure proved highly resilient to synthetic data generated near decision boundaries, accurately classifying minority pollution levels without overfitting.

## 🛠️ Technologies & Tools
* **Language:** Python
* **Core Libraries:** TensorFlow, Keras, PyCaret (Model Development), PyCM (Confusion Matrix & Multi-class evaluation).
* **Environment:** Google Colab, Anaconda (Jupyter Notebook).

## 📂 Project Directory Tree
```text
├── PM2.5 dataset/
│   └── Dataset/                      # Execution directory
│       ├── ET+SMOTE-checkpoint.ipynb # Main Jupyter Notebook (Preprocessing & Modeling)
│       └── ...                       # Secondary experimental scripts
├── .gitignore                        # Restricts raw datasets (>100MB) from version control
└── README.md                         # Detailed project summary documentation (This file)

👥 Contributors
Bachelor of Engineering in Computer Engineering, Mae Fah Luang University (Class of 2025):
•	Peerachet Khanitson
•	Chalermchai Nichee
•	Narawit Borunphan
•	Wisan Kittisaret
