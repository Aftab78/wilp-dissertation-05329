import azure.functions as func
import logging
import joblib
import json
import pandas as pd

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Load the model at the start so that it doesn't need to be loaded on every request
model = joblib.load('rfr_model.pkl')

@app.route(route="wilp_models", methods=["POST"])

def wilp_models(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Parse the request body
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid request body. Please pass valid JSON data.", 
            status_code=400
        )
    
    # Ensure that the necessary input data is present
    if 'data' not in req_body:
        return func.HttpResponse(
            "Please provide the input data in the request body as a 'data' field.", 
            status_code=400
        )

    try:
        # Assuming the input is a list of lists (for multiple predictions)
        input_data = req_body['data']
        df = pd.DataFrame(input_data)
        df['year'] = df['ALL_DATE'].apply(lambda x: int(x[-4:]))
        df['month'] = df['ALL_DATE'].apply(lambda x: int(x[3:5]))
        df['day'] = df['ALL_DATE'].apply(lambda x: int(x[:2]))  
        
        x = df[['year', 'month','day']]
        
        # Perform the prediction using the loaded model
        predictions = model.predict(x)

        # Convert the predictions into a JSON response
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
        
