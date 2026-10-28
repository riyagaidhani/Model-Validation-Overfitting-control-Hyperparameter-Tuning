
# House Price Prediction – Model Validation, Overfitting Control & Hyperparameter Tuning

**AI & ML Internship – Task 3 | Maincrafts Technology**

Extension of Task 2. This project goes beyond a single train-test split by detecting overfitting, validating models with cross-validation, and tuning hyperparameters with GridSearchCV on the California Housing dataset.

## Objective
- Detect and control overfitting
- Evaluate models reliably using 5-fold cross-validation
- Tune hyperparameters systematically with GridSearchCV
- Select the final model with a scientific justification

## Dataset
- **California Housing Dataset** (`sklearn.datasets.fetch_california_housing`)
- **Target:** Median House Value
- **Features:** Median income, house age, rooms, bedrooms, population, occupancy, latitude, longitude

## Tech Stack
Python, Jupyter Notebook, pandas, NumPy, scikit-learn, matplotlib, joblib

## Methodology
1. Loaded the dataset and split it into train (80%) and test (20%) sets (`random_state=42`)
2. Applied `StandardScaler` (fitted on training data only; inside a `Pipeline` during cross-validation to avoid data leakage)
3. Detected overfitting by comparing train vs test RMSE of an unconstrained Decision Tree
4. Evaluated Linear Regression, Ridge Regression and Decision Tree with 5-fold cross-validation
5. Tuned hyperparameters using GridSearchCV:
   - Decision Tree: `max_depth` [3, 5, 7, 10], `min_samples_split` [2, 5, 10]
   - Ridge: `alpha` [0.01, 0.1, 1, 10, 100]
6. Evaluated tuned models on the held-out test set (RMSE, R²)
7. Compared all models and selected the final one

## Results

| Model | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Linear Regression | 0.7205 | 0.7456 | 0.5758 |
| Ridge Regression (tuned) | 0.7205 | 0.7456 | 0.5758 |
| **Decision Tree (tuned)** | **0.6375** | **0.6454** | **0.6822** |

**Overfitting analysis:** Untuned Decision Tree – Train RMSE 0.0000 vs Test RMSE 0.7021 (gap 0.7021). After tuning, the gap reduced to 0.1650.

**Best parameters:** Decision Tree `max_depth=10, min_samples_split=10`, Ridge `alpha=0.01`

**Final model:** Tuned Decision Tree (lowest CV RMSE, lowest test RMSE and highest R²)



## Key Learnings
- High training accuracy means little if the model fails on new data
- Cross-validation gives a more trustworthy estimate than a single split
- GridSearchCV with constrained tree depth reduces overfitting
- Bias-variance tradeoff drives model choice
#Maincrafts Technology
