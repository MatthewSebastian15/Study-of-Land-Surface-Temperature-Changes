# Study of Land Surface Temperature Changes: Clustering (k-Means) and Time Series Prediction (XGBoost) 
This study analyzes global land surface temperature changes using K-Means clustering and time series prediction using XGBoost. K-Means clustering was employed to classify global temperature data into distinct categories (cold, moderate, hot) based on monthly surface temperature values. The XGBoost model was used to forecast temperature trends from 2025 to 2026 in selected countries with different climatic patterns. Evaluation metrics including the Silhouette Score and various regression error metrics (MAE, MSE, RMSE, MAPE) were used to measure clustering quality and prediction accuracy.

# Introduction
The rise of global land surface temperatures due to climate change presents a critical challenge. This research aims to uncover spatial and temporal patterns of temperature shifts from 1940 to 2024 using a dataset from Our World in Data. The approach involves two techniques:
1. Clustering via K-Means to identify homogeneous temperature regions.
2. Time series prediction via XGBoost to forecast temperatures in countries with distinct climate systems (e.g., Indonesia, Germany, Liberia, and the U.S.).

# Methodology
### A. Data Preparation
- Dataset Contains 198,120 rows with columns such as Average Monthly Temperature, Yearly Temperature, Anomalies, Entity, and Continent.
- Cleaning & Transformation:
  - Null values were dropped.
  - Label Encoding was applied to Entity and Continent.
  - Outliers (z-score > |3|) were removed.
  - Normality tests showed Temperature Anomaly as normally distributed; others were skewed.

### B. Data Visualization & EDA
- Pie charts revealed Africa as the most represented continent (29.9%).
- Histograms showed temperature variations across continents.
- Correlation matrix:
  - High correlation between monthly and yearly temperatures (r = 0.83).
  - Positive trend between year and anomaly values (r = 0.42).

# Clustering Techniques
### K-Means Clustering
- Initial Clustering (K=3)
  - Categories: Cold, Moderate, Hot.
  - Silhouette Score: 0.617
  - Hot clusters dominate Africa and Asia; Europe mostly Cold and Moderate.
- Improved Clustering (K=2)
  - Categories: Cold, Hot only.
  - Silhouette Score improved to 0.692, indicating better-defined clusters and less overlap.
  - Hot cluster dominated Africa, Asia, and America; Cold cluster dominated Europe.

# Time Series Prediction with XGBoost
XGBoost was used to forecast land surface temperature for 2025–2026, using historical monthly averages as training data. Evaluations were performed for countries with both tropical (2-season) and temperate (4-season) climates.

Model Evaluation (Selected Countries):
| Country   |  MAE  |  MSE  | RSME  | MAPE  |
| --------- | ----- | ----- | ----- | ----- |
| Indonesia | 0.151 | 0.035 | 0.189 | 0.006 |
| Liberia   | 0.235 | 0.088 | 0.297 | 0.009 |
| Germany   | 1.116 | 1.752 | 1.323 | 1.388 |
| U.S.      | 0.634 | 0.649 | 0.805 | 2.774 |

- Indonesia showed the highest prediction accuracy
- Climate complexity in Germany led to greater prediction errors
- Models for 2-season countries performed better overall

# Technologies and Tools
- **Language :** Python
- **Libraries :** Scikit-Learn, XGBoost, Pandas, NumPy
- **Visualization :** Matplotlib, Seaborn
- **Platform :** Jupyter Notebook

# Conclusion
K-Means clustering effectively identified meaningful global temperature categories, with K=2 giving the most compact results. The XGBoost model successfully predicted surface temperature trends, performing best in regions with simpler climatic patterns. Future work should include enhanced feature engineering, additional variables (e.g., elevation), and alternative models for comparative analysis.

Recommendations:
- Improve data balancing across continents.
- Use more robust outlier handling and normalization.
- Explore region-specific predictive models for enhanced accuracy.
