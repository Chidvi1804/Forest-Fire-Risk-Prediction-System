import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the processed data
data = pd.read_csv(r"C:\Users\csrid\OneDrive\Desktop\gda_proj\data\processed_data.csv")

# Split data into features and target
X = data.drop('y', axis=1)
y = data['y']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function for hyperparameter tuning and training
def train_and_tune_model(model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Define models and their hyperparameter grids
models = {
    "Logistic Regression": (LogisticRegression(max_iter=200), {'C': [0.1, 1, 10]}),
    "Support Vector Machine": (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    "Random Forest Classifier": (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    "Gradient Boosting Classifier": (GradientBoostingClassifier(), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]})
}

# Train models and evaluate accuracy
for model_name, (model, param_grid) in models.items():
    best_model = train_and_tune_model(model, param_grid)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"----- {model_name} -----")
    print(f"Best Parameters: {best_model.get_params()}")
    print(f"Accuracy: {accuracy:.2f}\n")
