import azure.functions as func
import logging
import joblib
import json
import pandas as pd
import XGBRegressor

# Load the model at the start so that it doesn't need to be loaded on every request
model1 = joblib.load('rfr_model.pkl')
model2 = joblib.load('svr_model.pkl')
model3 = XGBRegressor()
model3.load_model("xgbr_model.json")
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="rfr_models")
def rfr_models(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

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
        
        x = df[['year', 'month']]
        predictions = model1.predict(x)

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

@app.route(route="svr_models")
def svr_models(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

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
        
        x = df[['year', 'month']]
        predictions = model2.predict(x)

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
        
@app.route(route="xgbr_models")
def xgbr_models(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

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
        
        x = df[['year', 'month']]
        predictions = model3.predict(x)

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