from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Function to load model
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load Models
xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
extra_trees_model = load_model('et_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost-featureEnginered.pkl')

# Preprocess given data into a dataframe
def preprocess_data(customer_dict):
    # First, create the base input dictionary with exact feature names from training
    input_dict = {
        'CreditScore': customer_dict['CreditScore'],
        'Age': customer_dict['Age'],
        'Tenure': customer_dict['Tenure'],
        'Balance': customer_dict['Balance'],
        'NumOfProducts': customer_dict['NumOfProducts'],
        'HasCrCard': customer_dict['HasCreditCard'],  # Changed from HasCreditCard to HasCrCard
        'IsActiveMember': customer_dict['IsActiveMember'],
        'EstimatedSalary': customer_dict['EstimatedSalary'],
        'CLV': customer_dict['CLV'],
        'TenureAgeRatio': customer_dict['Tenure'] / customer_dict['Age'],  # Calculate TenureAgeRatio
        'Geography_Germany': customer_dict['Geography_Germany'],
        'Geography_Spain': customer_dict['Geography_Spain'],
        'Gender_Male': customer_dict['Gender_Male'],
        'AgeGroup_Middle-Aged': customer_dict['Middle-Aged'],  # Changed from Middle-Aged to AgeGroup_Middle-Aged
        'AgeGroup_Senior': customer_dict['Senior'],  # Changed from Senior to AgeGroup_Senior
        'AgeGroup_Elderly': customer_dict['Elderly']  # Changed from Elderly to AgeGroup_Elderly
    }
    
    customer_df = pd.DataFrame([input_dict])
    
    # Ensure columns are in the exact order expected by the model
    expected_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary', 'CLV', 'TenureAgeRatio',
        'Geography_Germany', 'Geography_Spain', 'Gender_Male',
        'AgeGroup_Middle-Aged', 'AgeGroup_Senior', 'AgeGroup_Elderly'
    ]
    
    customer_df = customer_df[expected_columns]
    return customer_df

# Get predictions and probabilities from all models and return average
def get_prediction(customer_dict):
    preprocessed_data = preprocess_data(customer_dict)

    # Get predictions from all models
    predictions = {
        'XGBoost': xgboost_model.predict(preprocessed_data)[0],
        'Naive Bayes': naive_bayes_model.predict(preprocessed_data)[0],
        'Random Forest': random_forest_model.predict(preprocessed_data)[0],
        'Decision Tree': decision_tree_model.predict(preprocessed_data)[0],
        'Extra Trees': extra_trees_model.predict(preprocessed_data)[0],
        'SVM': svm_model.predict(preprocessed_data)[0],
        'K-Nearest Neighbors': knn_model.predict(preprocessed_data)[0],
        'Voting Classifier': voting_classifier_model.predict(preprocessed_data)[0],
        'XGBoost-SMOTE': xgboost_SMOTE_model.predict(preprocessed_data)[0],
        'XGBoost-FeatureEngineered': xgboost_featureEngineered_model.predict(preprocessed_data)[0]
    }
    
    # Get probabilities from models that support predict_proba
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(preprocessed_data)[0][1],
        'Naive Bayes': naive_bayes_model.predict_proba(preprocessed_data)[0][1],
        'Random Forest': random_forest_model.predict_proba(preprocessed_data)[0][1],
        'Decision Tree': decision_tree_model.predict_proba(preprocessed_data)[0][1],
        'Extra Trees': extra_trees_model.predict_proba(preprocessed_data)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(preprocessed_data)[0][1],
        'XGBoost-SMOTE': xgboost_SMOTE_model.predict_proba(preprocessed_data)[0][1],
        'XGBoost-FeatureEngineered': xgboost_featureEngineered_model.predict_proba(preprocessed_data)[0][1]
    }
    
    return predictions, probabilities

# Endpoint to get predictions and probabilities
@app.post("/predict")
async def predict(data: dict):
    prediction, probabilities = get_prediction(data)
    return {
        "prediction": prediction.tolist(),
        "probability": probabilities.tolist()
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)