import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(probability=True),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier()
}

# Train models and store accuracy
accuracies = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    print(f"----- {model_name} -----")
    print(f"Accuracy: {accuracy:.2f}")

# Function for predicting fire
def predict_fire(model, new_data):
    new_data_df = pd.DataFrame([new_data], columns=X.columns)  # Convert to DataFrame with correct column names
    new_data_scaled = scaler.transform(new_data_df)  # Scale the new data
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)[:, 1]  # Get probability of class 1 (fire)
    return prediction[0], probability[0]

# Main function for user input
def main():
    while True:
        # Get user input for features
        try:
            X_new = []
            print("\nEnter the values for the following features:")
            features = X.columns.tolist()  # Use the same feature names as in the dataset
            for feature in features:
                value = float(input(f"{feature}: "))  # Ensure it's converted to float
                X_new.append(value)

            # Display predictions and probabilities for each model
            for model_name, model in models.items():
                prediction, probability = predict_fire(model, X_new)
                fire_status = "Fire" if prediction == 1 else "No Fire"
                print(f"{model_name}: Prediction: {fire_status}, Probability of Fire: {probability:.2f}, Accuracy: {accuracies[model_name]:.2f}")

        except ValueError:
            print("Invalid input. Please enter numerical values.")

        # Check if the user wants to continue
        continue_prompt = input("Do you want to enter another set of values? (yes/no): ").strip().lower()
        if continue_prompt != 'yes':
            break

if __name__ == "__main__":
    main()










