
# ============================================================
# SUPERKART SALES PREDICTION API
# ============================================================

# Import necessary libraries
import joblib
import pandas as pd

from flask import Flask, request, jsonify


# ------------------------------------------------------------
# Initialize Flask application
# ------------------------------------------------------------

superkart_sales_api = Flask("SuperKart Sales Prediction API")


# ------------------------------------------------------------
# Load the trained machine learning model
# ------------------------------------------------------------

# The serialized model contains the complete preprocessing
# pipeline along with the trained Random Forest model.
model = joblib.load("superkart_model.joblib")


# ------------------------------------------------------------
# Home endpoint
# ------------------------------------------------------------

@superkart_sales_api.get("/")
def home():
    """
    Handles GET requests to the root URL.

    Returns a simple message confirming that the
    SuperKart API is running.
    """

    return "Welcome to the SuperKart Sales Prediction API!"


# ------------------------------------------------------------
# Single prediction endpoint
# ------------------------------------------------------------

@superkart_sales_api.post("/v1/sales")
def predict_sales():
    """
    Handles POST requests for a single sales prediction.

    Expects a JSON payload containing the product and
    store characteristics required by the model.
    """

    try:

        # Get JSON data from the request body
        product_data = request.get_json()

        # Extract the features required by the model
        sample = {
            "Product_Id": product_data["Product_Id"],
            "Product_Weight": product_data["Product_Weight"],
            "Product_Sugar_Content": product_data["Product_Sugar_Content"],
            "Product_Allocated_Area": product_data["Product_Allocated_Area"],
            "Product_Type": product_data["Product_Type"],
            "Product_MRP": product_data["Product_MRP"],
            "Store_Id": product_data["Store_Id"],
            "Store_Establishment_Year": product_data["Store_Establishment_Year"],
            "Store_Size": product_data["Store_Size"],
            "Store_Location_City_Type": product_data["Store_Location_City_Type"],
            "Store_Type": product_data["Store_Type"]
        }

        # Convert the input into a Pandas DataFrame
        # so that it has the same structure expected by
        # the trained model pipeline.
        input_data = pd.DataFrame([sample])
        # --------------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------------

        # Derive Product ID Prefix from the first two characters
        # of Product_Id, as identified during EDA.
        input_data["Product_Id_Prefix"] = (
            input_data["Product_Id"].str[:2]
        )

        # Calculate Store Age from the establishment year.
        # 2026 is used as the reference year for deployment.
        input_data["Store_Age"] = (
            2026 - input_data["Store_Establishment_Year"]
        )

        # Normalize the inconsistent sugar-content category.
        # 'reg' and 'Regular' represent the same category.
        input_data["Product_Sugar_Content"] = (
            input_data["Product_Sugar_Content"]
            .replace({"reg": "Regular"})
        )


        # --------------------------------------------------------
        # Ensure columns are in exactly the same order expected
        # by the trained model.
        # --------------------------------------------------------
        model_features = [
            "Product_Id",
            "Product_Weight",
            "Product_Sugar_Content",
            "Product_Allocated_Area",
            "Product_Type",
            "Product_MRP",
            "Store_Id",
            "Store_Establishment_Year",
            "Store_Size",
            "Store_Location_City_Type",
            "Store_Type",
            "Product_Id_Prefix",
            "Store_Age"
        ]

        input_data = input_data[model_features]

        
        # Generate prediction
        predicted_sales = model.predict(input_data)[0]

        # Convert NumPy value to a standard Python float
        # so that Flask can serialize it as JSON.
        predicted_sales = round(float(predicted_sales), 2)

        # Return prediction
        return jsonify({
            "Predicted Product Store Sales Total": predicted_sales
        })

    except Exception as e:

        # Return an error message if the request cannot
        # be processed.
        return jsonify({
            "error": str(e)
        }), 400


# ------------------------------------------------------------
# Batch prediction endpoint
# ------------------------------------------------------------

@superkart_sales_api.post("/v1/salesbatch")
def predict_sales_batch():
    """
    Handles POST requests for batch sales prediction.

    Expects a CSV file containing multiple product-store
    records and returns predictions for all records.
    """

    try:

        # Get the uploaded CSV file from the request
        file = request.files["file"]

        # Read the CSV file into a Pandas DataFrame
        input_data = pd.read_csv(file)

        # Generate predictions for all rows
        predicted_sales = model.predict(input_data)

        # Convert predictions to Python floats
        predicted_sales = [
            round(float(sales), 2)
            for sales in predicted_sales
        ]

        # Add predictions to the input DataFrame
        output_data = input_data.copy()

        output_data["Predicted_Product_Store_Sales_Total"] = predicted_sales

        # Convert the result to a list of dictionaries
        # so it can be returned as JSON.
        output_dict = output_data.to_dict(orient="records")

        # Return predictions
        return jsonify({
            "predictions": output_dict
        })

    except Exception as e:

        # Return an error message if the batch request
        # cannot be processed.
        return jsonify({
            "error": str(e)
        }), 400


# ------------------------------------------------------------
# Run Flask application
# ------------------------------------------------------------

if __name__ == "__main__":

    superkart_sales_api.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
