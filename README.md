# 🌦️ Utilizing Big Data and Machine Learning for Air Pollution Monitoring: Air Quality Classification

An advanced machine learning and deep learning repository developed as an undergraduate Senior Project at **Mae Fah Luang University**. This project implements an end-to-end predictive and classification pipeline to monitor and categorize air quality levels based on meteorological conditions and pollutant concentrations.

---

## 📌 Project Objectives
* To analyze environmental and meteorological parameters affecting PM2.5 levels.
* To investigate the performance of various classification algorithms, feature scaling, and data transformations.
* To implement state-of-the-art oversampling techniques to effectively handle heavy class imbalance within air quality datasets.

## 📊 Dataset & Multi-Class Transformation
The project utilizes the **Beijing Multi-Site Air Quality Dataset** (spanning March 1, 2013, to February 28, 2017). Due to GitHub's file size limitations (100 MB maximum per file), raw and processed heavy `.csv` datasets are excluded from this repository and ignored via `.gitignore`.

* **Predictor Features:**
  * **6 Pollutants:** PM2.5, PM10, SO2, NO2, CO, O3
  * **Meteorological Factors:** Temperature (TEMP), Atmospheric Pressure (PRES), Dew Point (DEWP), Rainfall (RAIN), Wind Speed (WSPM), and Wind Direction (wd).
* **Feature Engineering Highlights:**
  * **Trigonometric Encoding:** Transformed the cyclic wind direction (`wd`) into continuous trigonometric components: `wd_sin` and `wd_cos` to properly retain spatial orientations.
  * **Target Labeling:** Transformed continuous PM2.5 numerical entries into a discrete **Five-Class Categorization** (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy) strictly aligned with the **Thai Air Quality Standard** issued by the Pollution Control Department (PCD), Thailand.

## ⚖️ Class Imbalance Management
To eliminate learning bias caused by heavily underrepresented air quality thresholds (specifically Levels 2 and 3), the dataset undergoes two primary oversampling strategies tested across five synthetic generation frameworks (SMOTE, Borderline-SMOTE, SMOTE-Tomek, ADASYN, and KMeans-SMOTE):
1. **Selective Oversampling:** Artificially expanding minority categories to 50% of the dominant majority class.
2. **Full Equalization:** Standardizing all class representations uniformly to build symmetric categorical boundaries.

## 🛠️ Project Directory Tree
```text
├── PM2.5 dataset/
│   └── Dataset/                      # Notebook directory for model execution
│       ├── ET+SMOTE-checkpoint.ipynb # Jupyter Notebook for Preprocessing & Core Modeling
│       └── ...                       # Secondary experimental scripts
├── .gitignore                        # Restricts huge datasets from version control tracking
└── README.md                         # Detailed project summary documentation (This file)
