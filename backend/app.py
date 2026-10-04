
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

    Expects a CSV file containing the following columns:

    Product_Weight
    Product_Sugar_Content
    Product_Allocated_Area
    Product_MRP
    Store_Size
    Store_Location_City_Type
    Store_Type
    Product_Id_char
    Store_Age_Years
    Product_Type_Category

    The endpoint converts the batch input columns into the
    feature names expected by the trained model pipeline.
    """

    try:

        # --------------------------------------------------------
        # Get uploaded CSV file
        # --------------------------------------------------------

        if "file" not in request.files:
            return jsonify({
                "error": "No CSV file was uploaded. Please use the 'file' field."
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No file selected."
            }), 400

        # --------------------------------------------------------
        # Read CSV
        # --------------------------------------------------------

        input_data = pd.read_csv(file)

        # --------------------------------------------------------
        # Required batch columns
        # --------------------------------------------------------

        required_columns = [
            "Product_Weight",
            "Product_Sugar_Content",
            "Product_Allocated_Area",
            "Product_MRP",
            "Store_Size",
            "Store_Location_City_Type",
            "Store_Type",
            "Product_Id_char",
            "Store_Age_Years",
            "Product_Type_Category"
        ]

        # Check for missing columns
        missing_columns = [
            column
            for column in required_columns
            if column not in input_data.columns
        ]

        if missing_columns:
            return jsonify({
                "error": "Missing required columns",
                "missing_columns": missing_columns
            }), 400

        # --------------------------------------------------------
        # Create model input
        # --------------------------------------------------------

        model_input = pd.DataFrame()

        model_input["Product_Weight"] = input_data["Product_Weight"]
        model_input["Product_Sugar_Content"] = input_data["Product_Sugar_Content"]
        model_input["Product_Allocated_Area"] = input_data["Product_Allocated_Area"]
        model_input["Product_Type"] = input_data["Product_Type_Category"]
        model_input["Product_MRP"] = input_data["Product_MRP"]

        # Product_Id_char in the batch file represents
        # the product ID prefix used by the model
        model_input["Product_Id_Prefix"] = input_data["Product_Id_char"]

        model_input["Store_Size"] = input_data["Store_Size"]
        model_input["Store_Location_City_Type"] = input_data["Store_Location_City_Type"]
        model_input["Store_Type"] = input_data["Store_Type"]

        # The model was trained using Store_Id as a categorical
        # feature, but the batch file does not contain Store_Id.
        #
        # We therefore provide a constant value for this feature.
        model_input["Store_Id"] = "UNKNOWN"

        # Store_Age_Years maps directly to Store_Age
        model_input["Store_Age"] = input_data["Store_Age_Years"]

        # --------------------------------------------------------
        # Generate predictions
        # --------------------------------------------------------

        predicted_sales = model.predict(model_input)

        predicted_sales = [
            round(float(sales), 2)
            for sales in predicted_sales
        ]

        # --------------------------------------------------------
        # Add predictions to original data
        # --------------------------------------------------------

        output_data = input_data.copy()

        output_data["Predicted_Product_Store_Sales_Total"] = predicted_sales

        # --------------------------------------------------------
        # Convert to JSON
        # --------------------------------------------------------

        output_dict = output_data.to_dict(orient="records")

        return jsonify({
            "predictions": output_dict
        })

    except Exception as e:

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
