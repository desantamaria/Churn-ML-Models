from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

with open('xgb_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def preprocess_data(customer_dict):
    
    input_dict = {
        'CreditScore': customer_dict['CreditScore'],
        'Age': customer_dict['Age'],
        "Tenure": customer_dict["Tenure"],
        "Balance": customer_dict["Balance"],
        "NumOfProducts": customer_dict["NumOfProducts"],
        "HasCreditCard": customer_dict["HasCreditCard"],
        "IsActiveMember": customer_dict["IsActiveMember"],
        "EstimatedSalary": customer_dict["EstimatedSalary"],
        "CLV": customer_dict["CLV"],
        "Middle-Aged": customer_dict['Middle-Aged'],
        "Senior": customer_dict['Senior'],
        "Elderly": customer_dict['Elderly'],
        "Geography_France": customer_dict['Geography_France'],
        "Geography_Germany": customer_dict['Geography_Germany'],
        "Geography_Spain": customer_dict['Geography_Spain'],
        "Gender_Male": customer_dict['Gender_Male']
    }
    
    customer_df = pd.DataFrame([input_dict])
    
    print("customer_df")
    print(customer_df)
    
    return customer_df

def get_prediction(customer_dict):
    preprocessed_data = preprocess_data(customer_dict)
    prediction = loaded_model.predict(preprocessed_data)
    probability = loaded_model.predict_proba(preprocessed_data)
    return prediction, probability

@app.post("/predict")
async def predict(data: dict):
    
    prediction, probabilities = get_prediction(data)
    
    return {
        "prediction":prediction.tolist(),
        "probability": probabilities.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

