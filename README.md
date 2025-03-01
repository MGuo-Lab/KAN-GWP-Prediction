# KAN-GWP-Prediction
A repository for an interpretable Global Warming Potential(GWP) prediction framework integrating molecular descriptors, process data using Kolmogorov–Arnold Networks (KAN).
This repository consists of three main components: the dataset used for modeling, the modeling part that includes AI model training and development, and the part dedicated to XAI analysis.


![fig1](https://github.com/user-attachments/assets/da6428a0-9096-4a10-8f97-2daf1939c876)
Fig. 1  Flowchart for Interpretable GWP Prediction: Feature Engineering, Model Comparison, Key Feature Identification, and White-box Modelling


Our methodology for GWP prediction integrates a multi-step framework, as illustrated in Fig. 1, encompassing feature engineering, model comparison, key feature identification, and white-box modelling. 

This systematic approach ensures both predictive accuracy and interpretability in addressing the environmental impact of chemical processes. The process begins with data collection and preprocessing, where LCI datasets are curated and refined to generate a robust training dataset. Key steps involve log transformation to address data skewness, consolidation of locational data, and exclusion of market-based entries to ensure fair model evaluation. 

Subsequently, feature engineering plays a pivotal role in extracting and optimizing chemical and process descriptors. Two distinct chemical feature sets, MACCS keys and Mordred descriptors, are employed to balance structural simplicity and physicochemical complexity. Additionally, process titles, descriptions, and locations are embedded using advanced natural language processing techniques, followed by dimensionality reduction through PCA to manage high-dimensional data effectively. 

The next stage, model comparison, involves benchmarking fundamental machine learning models, such as Random Forest (RF), XGBoost, DNN, and KAN. These models are evaluated using various feature combinations to determine the most effective predictive framework for GWP estimation. Following this, key feature identification is conducted using XAI techniques, specifically SHAP analysis. This step quantifies the contributions of chemical and process descriptors to model predictions, guiding the selection of an optimal feature set for interpretability. 

Finally, the framework transitions to white-box modelling, leveraging the KAN model to construct interpretable symbolic equations that capture both linear and non-linear relationships between features and GWP. This enables the derivation of a transparent, parameter-efficient model, achieving a balance between predictive performance and explainability.
