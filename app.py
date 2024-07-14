from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the initial model
def load_model():
    try:
        model = joblib.load('model_6.pkl')
        logging.info("Initial model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading initial model: {e}")
        model = None
    return model

model = load_model()
new_model = None

# Check if a new model exists
if os.path.exists('new_model.pkl'):
    try:
        new_model = joblib.load('new_model.pkl')
        logging.info("New model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading new model: {e}")
        new_model = None

# Preprocessing pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy='median'))
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_exclude=np.number))
)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive input data from the form
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        transaction_type = request.form['transactionType']

        type_payment = 1 if transaction_type == 'PAYMENT' else 0
        type_transfer = 1 if transaction_type == 'TRANSFER' else 0

        # Create a dictionary with the received data
        data = {
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'type_PAYMENT': type_payment,
            'type_TRANSFER': type_transfer
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Select columns to send to the model
        X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'type_PAYMENT', 'type_TRANSFER']]

        # Predict using the initial model
        if model:
            prob_old = model.predict_proba(X)[:, 1][0]
            fraud_status_old = "Fraud" if prob_old >= 0.5 else "Not Fraud"
            logging.info(f"Prediction with initial model: {prob_old}")
        else:
            prob_old = None
            fraud_status_old = "Model not loaded"
            logging.warning("Initial model is not loaded, cannot make prediction.")

        # Predict using the new model if available
        prob_new = None
        fraud_status_new = None
        if new_model:
            prob_new = new_model.predict_proba(X)[:, 1][0]
            fraud_status_new = "Fraud" if prob_new >= 0.5 else "Not Fraud"
            logging.info(f"Prediction with new model: {prob_new}")

        return render_template('result.html', prob_old=prob_old, prob_new=prob_new, fraud_status_old=fraud_status_old, fraud_status_new=fraud_status_new)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('error.html', error_message=str(e))

# Route to handle data upload and preprocessing
@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        # Assuming file input has a name 'file'
        file = request.files['file']
        if file:
            # Read the uploaded file
            df = pd.read_csv(file)
            logging.info("File uploaded successfully.")

            # Perform data preprocessing
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='unsigned')

            # Use category dtype for category column
            df['type'] = df['type'].astype('category')

            # Define features to remove
            features_to_remove = [0, 4, 6, 7, 8]

            # Perform data transformation and feature removal
            X_train, y_train, _ = data_transformations_feature_removal(df, features_to_remove)

            # Train a new model (BalancedRandomForestClassifier)
            global new_model
            new_model = BalancedRandomForestClassifier(max_depth=None, min_samples_leaf=1,
                                                       min_samples_split=5, n_estimators=500, n_jobs=-1,
                                                       random_state=42)
            new_model.fit(X_train, y_train)
            logging.info("New model trained successfully.")

            # Save the new model
            joblib.dump(new_model, 'new_model.pkl')
            logging.info("New model saved successfully.")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return render_template('error.html', error_message=str(e))

    return render_template('index.html', message="File uploaded and new model trained successfully.")

def data_transformations_feature_removal(data, features_to_remove):
    try:
        labels = None
        if 'isFraud' in data.columns:
            labels = data['isFraud']
            data = data.drop('isFraud', axis=1)
        if "nameOrig" in data.columns and "nameDest" in data.columns:
            data.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

        # Log the columns before preprocessing
        logging.info(f"Columns before preprocessing: {data.columns}")

        preprocessed_data = preprocessing.fit_transform(data)

        # Get feature names before removal
        features = preprocessing.get_feature_names_out()

        # Log the features
        logging.info(f"Features after preprocessing: {features}")

        # Remove features based on indices
        preprocessed_data = np.delete(preprocessed_data, features_to_remove, axis=1)

        # Update the list of features after removal
        remaining_features = np.delete(features, features_to_remove)

        if labels is not None:
            labels = labels.to_numpy()

        return preprocessed_data, labels, remaining_features

    except Exception as e:
        logging.error(f"Error during data transformation and feature removal: {e}")
        raise

if __name__ == "__main__":
    app.run(debug=True)
