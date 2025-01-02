# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Load and preprocess data
data = pd.read_csv(r"D:\gda_proj\data\processed_data_with_coords.csv")
X = data[['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'Latitude', 'Longitude']]
y = data['y']  # Target variable indicating fire risk

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models with tuned hyperparameters
log_reg = LogisticRegression(C=0.01, random_state=42)
svc = SVC(C=0.1, kernel='rbf', probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42)
gb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=3, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.01, max_depth=3, n_estimators=100, random_state=42)

# Ensemble: Voting Classifier with weighted voting based on model F1 scores
weighted_voting_clf = VotingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('svc', svc),
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb)
    ],
    voting='soft',
    weights=[1, 2, 1, 3, 2]  # Adjusted based on performance
)

# Stacking Classifier using Logistic Regression as the meta-model
stacking_clf = StackingClassifier(
    estimators=[
        ('svc', svc),
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# Fit models and evaluate on test data
models = {
    'Logistic Regression': log_reg,
    'Support Vector Machine': svc,
    'Random Forest Classifier': rf,
    'Gradient Boosting Classifier': gb,
    'XGBoost Classifier': xgb,
    'Weighted Voting Classifier': weighted_voting_clf,
    'Stacking Classifier': stacking_clf
}

# Train each model, make predictions, and display F1 score
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score for {name}: {f1:.2f}")

# User input for prediction
def get_user_input():
    print("\nEnter the values for the following features:")
    features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'Latitude', 'Longitude']
    values = [float(input(f"{feature}: ")) for feature in features]
    return np.array([values])

# Function to predict with all models
def make_predictions(input_data):
    scaled_data = scaler.transform(input_data)
    for name, model in models.items():
        prob_fire = model.predict_proba(scaled_data)[0][1] if hasattr(model, 'predict_proba') else None
        prediction = model.predict(scaled_data)
        prediction_text = "Fire" if prediction[0] == 1 else "No Fire"
        prob_text = f", Probability of Fire: {prob_fire:.2f}" if prob_fire is not None else ""
        print(f"{name}: Prediction: {prediction_text}{prob_text}")

# User interaction
while True:
    input_data = get_user_input()
    make_predictions(input_data)
    cont = input("Do you want to enter another set of values? (yes/no): ").strip().lower()
    if cont != 'yes':
        break
