# Bogotá Air Quality Forecasting

This repository contains a comprehensive study and implementation of time-series forecasting models to predict daily **Fine Particulate Matter (PM2.5)** levels in Bogotá, Colombia, with a 7-day horizon.

The project evolves through **three distinct phases**, moving from rigorous exploratory analysis to classical machine learning benchmarks, and finally, to **State-of-the-Art (SOTA)** deep learning architectures. The goal is to build a reliable forecasting system crucial for public health warnings and environmental policy.

**Authors:**
- Alejandra Valle Fernandez
- [Juan Guillermo Gómez](https://www.linkedin.com/in/jggomezt/)

---

## 📂 Project Phases Overview

### **Phase 1: EDA, Stationarity & Baselines**
The foundation of the project. This phase focuses on understanding the data structure, identifying patterns (trend/seasonality), applying transformations to achieve stationarity, and establishing a performance baseline using naive models.

### **Phase 2: Statistical & Traditional ML Benchmark**
Implementation of classical statistical methods and tree-based Machine Learning models using `statsforecast` and `mlforecast`.
* **Key Techniques:** Manual & Auto ARIMA, Exponential Smoothing, Feature Engineering (Lags, Rolling Windows).
* **Optimization:** Hyperparameters tuned using **Optuna**.

### **Phase 3: SOTA Deep Learning Architectures**
Implementation of **State-of-the-Art (SOTA)** Neural Forecasting models using the `nixtla` ecosystem (`neuralforecast`).
* **Why SOTA?** We utilize cutting-edge architectures (Transformers, MLP-Mixers) that currently hold the highest performance records in global time-series benchmarks, capable of modeling complex non-linearities and long-term dependencies better than traditional methods.
* **Optimization:** Hyperparameters tuned using **Ray**.

---

## 📊 Dataset Characterization

The analysis utilizes the **Latin America Weather and Air Quality Data** (Source: [Kaggle](https://www.kaggle.com/datasets/anycaroliny/latin-america-weather-and-air-quality-data)), derived from the Open-Meteo API.

* **Location:** Bogotá, Colombia.
* **Target Variable:** `pm2_5` (Particulate Matter 2.5 concentration in µg/m³).
* **Frequency:** Daily.
* **Observation Period:** August 2022 - April 2024 (Continuous).

### Statistical Summary (Bogotá PM2.5)

| Statistic | Value | Interpretation |
|---|---|---|
| **Mean** | 21.55 µg/m³ | Average daily concentration. |
| **Max (Capped)** | 46.66 µg/m³ | Significant pollution peaks observed. |
| **CV** | 42.07% | High variability relative to the mean. |
| **Distribution** | Right-skewed | Occasional extreme events drive the skewness. |
| **Stationarity** | **Non-Stationary** | Confirmed by ADF and KPSS tests (p-values indicate strong trend). |

---

## 🔬 Phase 1: Detailed Analysis & Findings

Before modeling, a rigorous statistical analysis was conducted to determine the optimal preprocessing steps.

### 1. Stationarity & Transformations
We tested multiple transformations using **Augmented Dickey-Fuller (ADF)** and **KPSS** tests.

* **Original Series:** Non-stationary due to trend.
* **Winner:** **Logarithmic Transformation + First Difference ($d=1$)**.
    * *Result:* This combination achieved the lowest ADF statistic (-11.879) and successfully stabilized both variance and mean.

### 2. Seasonal Decomposition
* **Model:** Multiplicative decomposition proved superior to additive.
* **Insight:** A clear **Trend** component exists, but **Seasonality is weak** (weekly cycle strength ~0.003).

### 3. SARIMA Parameter Identification
Based on ACF/PACF plots of the transformed series, the theoretical structure for ARIMA models was identified as:
* **AR ($p$):** 3 (Sharp cutoff at Lag 3)
* **MA ($q$):** 2 (Sharp cutoff at Lag 2)
* **Diff ($d$):** 1 (Required for stationarity)
* **Seasonal:** No significant seasonal AR/MA terms found ($P=0, Q=0$).

### 4. Baseline Results
We evaluated naive models (Drift, Naive, Seasonal Naive, SMA).
* **Best Baseline:** **Drift Model with Logarithmic Differencing**.
* **Benchmark RMSE:** **8.04 µg/m³** (This is the score subsequent phases aim to beat).

### 5. Results
<img width="1244" height="900" alt="newplot" src="https://github.com/user-attachments/assets/65825eff-0747-4cc2-9384-227bbb0b0aa6" />

<img width="1244" height="900" alt="newplot (1)" src="https://github.com/user-attachments/assets/c760a314-5d4c-4ac6-9333-1f7bd4ae3a73" />

<img width="1244" height="900" alt="newplot (2)" src="https://github.com/user-attachments/assets/88e997f8-2a93-475e-9ddc-02e91a248cb4" />

<img width="1244" height="1000" alt="newplot (3)" src="https://github.com/user-attachments/assets/d4ca948a-d42a-49e3-b210-7d1aafce6e7f" />

<img width="1244" height="1000" alt="newplot (4)" src="https://github.com/user-attachments/assets/014afb15-32e4-456a-bcf4-99ee4035206d" />

<img width="1800" height="1800" alt="newplot (5)" src="https://github.com/user-attachments/assets/9f1dcbc6-c3bc-4985-a393-c27945c178a0" />

<img width="1244" height="400" alt="newplot (6)" src="https://github.com/user-attachments/assets/07a23381-fc64-46cc-b78d-e936ceea6ea4" />

<img width="1244" height="500" alt="newplot (7)" src="https://github.com/user-attachments/assets/b8a8255d-424f-4107-89b5-d85122393ba1" />

<img width="1244" height="600" alt="newplot (8)" src="https://github.com/user-attachments/assets/0098c319-bb5d-45d9-8736-2d9d47d91bec" />

---

## 🤖 Phase 2: Statistical & Machine Learning Models

In this phase, we moved beyond baselines to robust statistical and ML approaches.

### Models Implemented
1.  **Statistical Family (`statsforecast`):**
    * **ARIMA(3,1,2):** Manually configured based on Phase 1 findings.
    * **AutoARIMA:** Automatic grid search for optimal $(p,d,q)$.
    * **SARIMA(3,1,2)x(1,0,0)[7]:** Testing weekly seasonality explicitly.
    * **Holt & HoltWinters:** Exponential smoothing for trend/seasonality.

2.  **Machine Learning Family (`mlforecast`):**
    * **Regressors:** `LinearRegression`, `XGBoost`, `LightGBM`, `RandomForest`.
    * **Feature Engineering:**
        * Lags: $t-1$ to $t-7$.
        * Rolling Means: Windows of 3 and 7 days.
    * **Optimization:** Hyperparameters tuned using **Optuna**.

### Results

<img width="1244" height="500" alt="newplot" src="https://github.com/user-attachments/assets/d5bd166a-58b7-47d7-b36c-e21d43f90c22" />

<img width="1232" height="700" alt="newplot (1)" src="https://github.com/user-attachments/assets/419ad0e1-2e94-4691-8fae-7d5b6dcd3051" />

<img width="1232" height="700" alt="newplot (2)" src="https://github.com/user-attachments/assets/4641e5b6-5ed1-467a-b485-3059b239e70c" />

<img width="1244" height="500" alt="newplot (3)" src="https://github.com/user-attachments/assets/7aa1f711-5e80-4ebf-9d40-0f5201a20708" />

<img width="1200" height="600" alt="newplot (4)" src="https://github.com/user-attachments/assets/10953d2e-fa26-4053-9301-e72abcd011e1" />

<img width="1244" height="700" alt="newplot (5)" src="https://github.com/user-attachments/assets/89f0f2c6-8b75-45c9-93e6-bb6953450abf" />

<img width="1244" height="700" alt="newplot (6)" src="https://github.com/user-attachments/assets/4de654d7-a97d-40ed-89a0-f16f7c978eb3" />

<img width="1244" height="500" alt="newplot (7)" src="https://github.com/user-attachments/assets/b5ecfff1-6f5e-4e78-9581-87174d2dc830" />

<img width="1200" height="600" alt="newplot (8)" src="https://github.com/user-attachments/assets/7ab14219-ef21-4666-ad4b-9cc68b839b09" />

<img width="1244" height="700" alt="newplot (9)" src="https://github.com/user-attachments/assets/0367ed4c-08fe-431f-a2bc-e0021f8429b0" />

<img width="1244" height="700" alt="newplot (10)" src="https://github.com/user-attachments/assets/fbf15696-43cb-4676-bdb9-829d9731af86" />

<img width="1244" height="700" alt="newplot (11)" src="https://github.com/user-attachments/assets/dea83adf-577a-40ed-8b40-ed7dd54252e7" />

<img width="1244" height="700" alt="newplot (12)" src="https://github.com/user-attachments/assets/16b0d42d-3785-4ff5-9bfb-dcab5906ac41" />

<img width="1244" height="700" alt="newplot (13)" src="https://github.com/user-attachments/assets/e4cad58e-b670-4b4a-8e83-f67cacc99480" />

<img width="1244" height="700" alt="newplot (14)" src="https://github.com/user-attachments/assets/c76040ff-7e7e-486a-9b5b-edc20a37847a" />

<img width="1244" height="600" alt="newplot (15)" src="https://github.com/user-attachments/assets/2002af5b-5189-452e-92aa-9ff893b71694" />

<img width="1000" height="600" alt="newplot (16)" src="https://github.com/user-attachments/assets/eb2d373a-552e-42e2-8eaf-1222141d2c2c" />

---

## 🧠 Phase 3: SOTA Deep Learning Architectures

The final phase leverages **State-of-the-Art (SOTA)** Deep Learning.

### What does "SOTA" mean in this context?
State-of-the-Art models represent the current pinnacle of time series research (2019-2024). Unlike traditional RNNs, these architectures address specific limitations, such as long-sequence volatility and computational inefficiency. They are designed to:
* **Capture Global Dependencies:** Using attention mechanisms and decomposition (Trend vs. Seasonality) within the network layers.
* **Handle Exogenous Variables:** Seamlessly integrating weather data (`pm10`, `CO`, `NO2`) to improve prediction accuracy.
* **Prevent Overfitting:** Using advanced block structures (like stacks in N-BEATS).

### Models Implemented (`neuralforecast`)
We utilized the `Auto` class for optimized architecture search, leveraging **Ray** for parallel hyperparameter tuning:

* **MLP-Based (Efficient & Powerful):**
    * **AutoTiDE (Time-series Dense Encoder):** An MLP-based encoder-decoder model that outperforms Transformers in speed and accuracy for many tasks.
    * **AutoNBEATS & AutoNHITS:** Architectures based on backward/forward residual links, excellent for interpretable trend/seasonality decomposition.
* **Transformers & Modern Convolutions:**
    * **AutoTimesNet:** A SOTA model that transforms 1D time series into 2D tensors to capture multi-periodicity.
    * **AutoTCN, AutoAutoformer, AutoInformer:** Advanced sequence modeling.

### Results

<img width="996" height="547" alt="img" src="https://github.com/user-attachments/assets/96ce7024-ce7a-4fab-baf8-15299fa94fdd" />

<img width="1151" height="624" alt="img2" src="https://github.com/user-attachments/assets/36933b34-2a78-4d68-859c-5dd2c6d5e53a" />

<img width="1000" height="700" alt="newplot" src="https://github.com/user-attachments/assets/ecf13189-aefb-42f0-ad61-565c2d7bc491" />

<img width="1000" height="700" alt="newplot (1)" src="https://github.com/user-attachments/assets/3824a6b9-a13a-4bf4-9f28-b64b9144686e" />

<img width="1244" height="900" alt="newplot (2)" src="https://github.com/user-attachments/assets/b963a95c-0650-42a5-b698-e5a7569c9696" />

<img width="996" height="547" alt="img3" src="https://github.com/user-attachments/assets/d03bddc8-6752-44df-9844-4003b3c7c62c" />

<img width="1151" height="624" alt="img4" src="https://github.com/user-attachments/assets/4a67e693-555b-42b1-bbc1-257daf29b21a" />

<img width="1000" height="700" alt="newplot (3)" src="https://github.com/user-attachments/assets/816e0f1e-0c36-43e3-89e8-009843b552e5" />

<img width="1000" height="700" alt="newplot (4)" src="https://github.com/user-attachments/assets/20b81834-6f48-4b99-9fb2-a03120ee395c" />

<img width="1244" height="900" alt="newplot (5)" src="https://github.com/user-attachments/assets/8c38aee3-72a4-40fc-8caa-76b74896c934" />

<img width="996" height="547" alt="img5" src="https://github.com/user-attachments/assets/7e9b6108-4fe4-42c4-a5f9-970b21521682" />

<img width="1151" height="624" alt="img6" src="https://github.com/user-attachments/assets/33dae7a2-94c7-4759-9b2b-112ee3e25e3a" />

<img width="1000" height="700" alt="newplot (6)" src="https://github.com/user-attachments/assets/7dba18d5-2cc3-459b-89b5-2b132e26f966" />

<img width="1000" height="700" alt="newplot (7)" src="https://github.com/user-attachments/assets/fea13754-6ce7-417d-875b-286177a88cb5" />

<img width="1244" height="900" alt="newplot (8)" src="https://github.com/user-attachments/assets/f6c8d494-94bb-4814-b6c0-4c5cfc0fa3f6" />

<img width="1009" height="547" alt="img7" src="https://github.com/user-attachments/assets/8fe0aa8c-cbfd-47dc-a1b9-977ea2e76171" />

<img width="1151" height="624" alt="img8" src="https://github.com/user-attachments/assets/8f3470ab-d455-47ea-bd5c-d8e514a13d59" />

<img width="1000" height="700" alt="newplot (9)" src="https://github.com/user-attachments/assets/875f73aa-46f5-4b24-b330-c69c32e1f9a2" />

<img width="1000" height="700" alt="newplot (10)" src="https://github.com/user-attachments/assets/5a37a4ff-4b73-4a5d-b7d1-c92bf36a8d40" />

<img width="1244" height="900" alt="newplot (11)" src="https://github.com/user-attachments/assets/11871b32-9a0c-4713-a27c-e03eaab816cd" />

<img width="1244" height="900" alt="newplot (12)" src="https://github.com/user-attachments/assets/45a41fbb-aa76-47af-a202-bbfc724d95ff" />

---

## Final Report

<img width="1232" height="1000" alt="newplot (13)" src="https://github.com/user-attachments/assets/f6f03ff2-5dcd-4d4b-97e9-e20509fabd3f" />

Global Conclusions for all models Statistical, Machine Learning, and Deep Learning:

### **1. Strategic Analysis: Stability vs. Reactivity**

* **The "SOTA Neutral Model" (AutoInformer):**
    * **Metrics Winner:** AutoInformer has achieved the State-of-the-Art (SOTA) position with the lowest global error (**RMSE 7.58**) and robust neutrality (**Bias 0.25**).
    * **Behavior:** It acts as a "Noise Filter." By utilizing sparse attention mechanisms, it successfully separates the signal from the noise in a small dataset (~500 samples), providing the most reliable central tendency forecast.
    * **The Advantage:** It is the only model that combines high precision (MAE 6.09) with a lack of systematic bias, making it the most trustworthy model for general planning.

* **The "Pessimistic Optimizer" (AutoAutoformer):**
    * **High Accuracy, High Bias:** Surprisingly, it ranks **#2** in accuracy (RMSE 7.62) but exhibits a severe **negative bias (-2.46)**.
    * **Behavior:** It minimizes error by staying conservative. It systematically underestimates demand or pollution peaks. While "safe" for minimizing overstock, it is dangerous for risk management (missing alerts).

* **The "Reactive Baselines" (AutoTiDE and ARIMA):**
    * **AutoTiDE:** Although it ranks #5, it maintains neutrality (Bias 0.25) similar to the Informer but with slightly less precision. It remains valuable for its ability to incorporate exogenous variables directly via MLPs.
    * **ARIMA:** Remains the undisputed king of "Small Data" statistical baselines (#3), beating complex Deep Learning models like TCN and NBEATS.

---

### **2. Use Cases: Which model to choose based on the objective?**

* **Case A: General Forecasting the "Gold Standard"**
    * **Objective:** Accurate forecasting where cumulative totals matter.
    * **Recommendation:** **AutoInformer**.
    * **Why:** Its bias near zero (0.25) ensures that over-predictions and under-predictions cancel each other out over time. It offers the best fidelity to the average reality of the phenomenon.

* **Case B: Risk Management and Peak Detection**
    * **Objective:** Detect critical high values (e.g., pollution spikes) to trigger safety protocols.
    * **Recommendation:** **LSTM** or **ARIMA**.
    * **Why:**
        * **LSTM:** showed "High Fidelity" to non-linear dynamics and volatility.
        * **ARIMA:** reacts quickly to inertial rises.
        * *Avoid:* AutoAutoformer, as it significantly underestimates peaks (Bias -2.46), leading to missed alerts.

* **Case C: Conservative/Minimum Estimation**
    * **Objective:** Ensure we cover the "guaranteed" demand without overcommitting resources.
    * **Recommendation:** **AutoAutoformer**.
    * **Why:** Its specific tendency to under-forecast negative bias makes it a natural "lower bound" estimator.

---

### **3. Consolidated Table**

Ordered by **RMSE** from best to worst general accuracy, including the behavior profile for decision-making.

| Rank | Category | Model | RMSE | MAE | Bias | Behavior Profile | Ideal Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 **1** | **Deep Learning** | **AutoInformer** | **7.58** | **6.09** | **0.25** | **Neutral & Robust.** Excellent calibration. SOTA accuracy. | General Planning, Base Stock, Budgeting. |
| 🥈 **2** | Deep Learning | AutoAutoformer | 7.62 | 6.30 | **-2.46** | **Pessimistic.** Systematically underestimates peaks. | Risk-averse conservative planning. |
| 🥉 **3** | **Statistical** | **ARIMA** | **8.78** | 7.18 | -- | **Inertial.** Reacts quickly to recent history. | **Immediate Alerts**, Short term forecasting. |
| 4 | Statistical | SARIMA | 8.81 | 7.17 | -- | **Seasonal Precision.** Captures weekly cycles well. | Operations requiring daily precision. |
| 5 | Deep Learning | AutoTiDE | 8.85 | 6.99 | 0.25 | **Reactive & Balanced.** Uses exogenous variables well. | Variable demand, Trend detection. |
| 6 | Deep Learning | LSTM | 8.90 | 7.00 | 0.10 | **High Fidelity.** Captures volatility well. | Complex, unstable series. |
| 7 | Deep Learning | AutoTimesNet | 9.18 | 6.86 | -0.61 | **Underfitted.** Fails to capture trends (Lag 1 issues). | Detecting local valleys only. |
| 8 | Statistical | Holt (Exp.) | 9.63 | 7.73 | -- | **Smoother.** Lags behind fast changes. | Robust baseline for noisy data. |
| 9 | Statistical | HoltWinters | 9.65 | 7.75 | -- | **Seasonal Smoother.** Rigid structure. | Strictly periodic stable data. |
| 10 | Classic ML | XGBRegressor | 9.74 | 7.78 | -- | **Bounded.** Cannot extrapolate trends. | Tabular regression (Interpolation). |
| 11 | Classic ML | RandomForest | 9.77 | 7.68 | -- | **Average/Stable.** Low variance but poor trend detection. | Stationary noisy data. |
| 12 | Classic ML | Ridge | 9.88 | 7.76 | -- | **Linear.** Assumes simple relationships. | Simple linear baselines. |
| 13 | Classic ML | LGBMRegressor | 10.00 | 8.04 | -- | **Fast Boosting.** Overfitted on small data. | Large-scale tabular data. |
| 14 | Classic ML | Lasso | 10.06 | 8.03 | -- | **Feature Selector.** Aggressive regularization lost signal. | Not recommended. |
| 15 | Deep Learning | DeepAR | 10.27 | 8.07 | -1.83 | **Probabilistic.** Pessimistic range estimation. | Uncertainty estimation (Inventory risk). |
| 16 | Classic ML | KNeighbors | 10.44 | 8.39 | -- | **Distance-based.** Fails on trends. | **Not recommended.** |
| 17 | Deep Learning | AutoNBEATS | 10.73 | 8.49 | -0.83 | **Noisy/Biased.** Leaks signal (Bad ACF). | Failed to decompose signal correctly. |
| 18 | Deep Learning | AutoTCN | 10.97 | 8.85 | 0.06 | **Unstable.** Heavy tails and high error. | **Not recommended.** Too much noise. |

---

### **4. Final Deployment Recommendation**

* **Primary Model:** Deploy **AutoInformer**. It has finally surpassed the statistical baselines and offers the most accurate and unbiased representation of the time series.
* **Secondary/Fallback:** Maintain **ARIMA** as a lightweight fallback system. It is computationally cheap and proved to be more robust than complex DL models like TCN or NBEATS in this specific "Small Data" scenario.

---

## 🛠 Technology Stack

* **Core:** `pandas`, `numpy`, `scikit-learn`, `statsmodels.`
* **Forecasting Ecosystem:**
    * `statsforecast` (Fast ARIMA/ETS)
    * `mlforecast` (ML applied to time series)
    * `neuralforecast` (PyTorch-based Deep Learning)
    * `utilsforecast` & `window-ops`
* **Optimization:**
    * `optuna` (Phase 2)
    * `ray[tune]` (Phase 3 - Deep Learning)
* **Visualization:** `matplotlib`, `seaborn`, `plotly.`
* **Utilities:** `rich`

---

## 📚 References & Further Reading

The implementation of SOTA Deep Learning architectures in Phase 3 is based on the following research:

* **TiDE:** [Time Series Forecasting with TiDE](https://medium.com/the-forecaster/time-series-forecasting-with-tide-b043acc60f79)
* **N-BEATS:** [Extend N-BEATS for Accurate Time Series Forecasting](https://medium.com/the-forecaster/extend-n-beats-for-accurate-time-series-forecasting-0f78427b45a9)
* **N-HiTS:** [All About N-HiTS: The Latest Breakthrough in Time Series Forecasting](https://medium.com/towards-data-science/all-about-n-hits-the-latest-breakthrough-in-time-series-forecasting-a8ddcb27b0d5)
* **TimesNet:** [TimesNet: The Latest Advance in Time Series Forecasting](https://www.datasciencewithmarco.com/blog/timesnet-the-latest-advance-in-time-series-forecasting)
* **LSTM:** [Long Short-Term Memory (LSTM) Networks](https://medium.com/@RobuRishabh/long-short-term-memory-lstm-networks-9f285efa377d)
* **DeepAR:** [DeepAR: Mastering Time Series Forecasting with Deep Learning](https://towardsdatascience.com/deepar-mastering-time-series-forecasting-with-deep-learning-bc717771ce85/)
* **Autoformer:** [Autoformer: Decomposing the Future of Time Series Forecasting](https://medium.com/@kdk199604/autoformer-decomposing-the-future-of-time-series-forecasting-e082446eab8f)
* **Informer:** [Informer: Beyond Efficient Transformer for Long Sequence Time Series Forecasting](https://rezayazdanfar.medium.com/informer-beyond-efficient-transformer-for-long-sequence-time-series-forecasting-4eeabb669eb)
