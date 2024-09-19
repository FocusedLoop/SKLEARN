import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Generate the regression data
X, y = make_regression(n_samples=10000, n_features=8, n_informative=2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# Create the XGBRegressor model with GPU support
xgb_reg = XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda')

# Define the hyperparameter grid with appropriate XGBoost parameters
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # Number of trees
    'max_depth': [10, 20, 30, 40, None],  # Maximum depth of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate (eta)
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used per tree
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used per tree
    'min_child_weight': [1, 2, 3]  # Minimum sum of instance weight for a child
}

# Perform grid search
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Evaluate the model
print(f"Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Predict
specific_input = scaler.transform([[1, 2, 3, 4, 5, 6, 7, 8]])
specific_prediction = best_model.predict(specific_input)
print(f"Prediction for input: {specific_prediction}")

