import azure.functions as func
import logging
import joblib
import json
import pandas as pd

# Load the model at the start so that it doesn't need to be loaded on every request
rfr_model = joblib.load('rfr_model.pkl')
xgbr_model = joblib.load('expert_model.pkl')

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


def predict(req, model):

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid request body. Please pass valid JSON data.", 
            status_code=400
        )

    if 'data' not in req_body:
        return func.HttpResponse(
            "Please provide the input data in the request body as a 'data' field.", 
            status_code=400
        )

    try:
        input_data = req_body['data']
        df = pd.DataFrame(input_data)
        df['year'] = df['ALL_DATE'].apply(lambda x: int(x[-4:]))
        df['month'] = df['ALL_DATE'].apply(lambda x: int(x[3:5]))
        #df['day'] = df['ALL_DATE'].apply(lambda x: int(x[:2]))  
        
        x = df[['year', 'month']]
        predictions = model.predict(x)

        result = {
            "predictions": predictions.tolist()
        }

        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return func.HttpResponse(
            "An error occurred while performing prediction. Please check your input data and try again.",
            status_code=500
        )
# Route for Random Forest Model
@app.route(route="predict-rfr", methods=["POST"])
def predict_rfr(req: func.HttpRequest) -> func.HttpResponse:
    return predict(req, rfr_model)

# Route for XGBoost Model
@app.route(route="predict-xgbr", methods=["POST"])
def predict_xgbr(req: func.HttpRequest) -> func.HttpResponse:
    return predict(req, xgbr_model)