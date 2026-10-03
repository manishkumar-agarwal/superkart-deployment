
# ============================================================
# SUPERKART SALES PREDICTION - STREAMLIT FRONTEND
# ============================================================

import streamlit as st
import pandas as pd
import requests


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)


# ------------------------------------------------------------
# Backend API URL
# ------------------------------------------------------------

# During local Docker testing, the backend container will be
# accessible using its Docker service/container name.
#
# This can later be changed to the forwarded Codespaces URL
# for online inference.

BACKEND_URL = "http://backend:7860"


# ------------------------------------------------------------
# Application title
# ------------------------------------------------------------

st.title("🛒 SuperKart Sales Prediction")

st.write(
    """
    Use this application to predict the expected sales revenue
    for a product-store combination.
    """
)


# ------------------------------------------------------------
# Create tabs for single and batch prediction
# ------------------------------------------------------------

single_tab, batch_tab = st.tabs(
    ["Single Prediction", "Batch Prediction"]
)


# ============================================================
# SINGLE PREDICTION
# ============================================================

with single_tab:

    st.header("Single Sales Prediction")

    # --------------------------------------------------------
    # Product information
    # --------------------------------------------------------

    st.subheader("Product Information")

    col1, col2 = st.columns(2)

    with col1:

        product_id = st.text_input(
            "Product ID",
            value="FD6114"
        )

        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=12.66,
            step=0.01
        )

        product_sugar_content = st.selectbox(
            "Product Sugar Content",
            ["Low Sugar", "Regular", "No Sugar"]
        )

        product_allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            value=0.027,
            step=0.001,
            format="%.3f"
        )

        product_type = st.selectbox(
            "Product Type",
            [
                "Baking Goods",
                "Breads",
                "Breakfast",
                "Canned",
                "Dairy",
                "Frozen Foods",
                "Fruits and Vegetables",
                "Hard Drinks",
                "Health and Hygiene",
                "Household",
                "Meat",
                "Others",
                "Seafood",
                "Snack Foods",
                "Soft Drinks",
                "Starchy Foods"
            ]
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=117.08,
            step=0.01
        )

    with col2:

        store_id = st.selectbox(
            "Store ID",
            ["OUT001", "OUT002", "OUT003", "OUT004"]
        )

        store_establishment_year = st.selectbox(
            "Store Establishment Year",
            [1987, 1998, 1999, 2009]
        )

        store_size = st.selectbox(
            "Store Size",
            ["Small", "Medium", "High"]
        )

        store_location_city_type = st.selectbox(
            "Store Location City Type",
            ["Tier 1", "Tier 2", "Tier 3"]
        )

        store_type = st.selectbox(
            "Store Type",
            [
                "Departmental Store",
                "Supermarket Type1",
                "Supermarket Type2",
                "Food Mart"
            ]
        )


    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "Predict Sales",
        type="primary"
    ):

        # Create request payload
        payload = {
            "Product_Id": product_id,
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Establishment_Year": store_establishment_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type
        }

        try:

            # Send request to Flask backend
            response = requests.post(
                f"{BACKEND_URL}/v1/sales",
                json=payload,
                timeout=30
            )

            # Check whether request was successful
            response.raise_for_status()

            # Extract prediction
            result = response.json()

            predicted_sales = result[
                "Predicted Product Store Sales Total"
            ]

            # Display prediction
            st.success(
                f"Predicted Sales Revenue: "
                f"{predicted_sales:,.2f}"
            )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to the backend API: {e}"
            )

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# ============================================================
# BATCH PREDICTION
# ============================================================

with batch_tab:

    st.header("Batch Sales Prediction")

    st.write(
        """
        Upload a CSV file containing multiple product-store
        records. The file should contain the same predictor
        columns used by the trained model.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Read uploaded CSV
        input_data = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.subheader("Uploaded Data")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Batch prediction button
        # ----------------------------------------------------

        if st.button(
            "Generate Batch Predictions",
            type="primary"
        ):

            try:

                # Reset file pointer before sending
                uploaded_file.seek(0)

                # Send CSV file to Flask backend
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/v1/salesbatch",
                    files=files,
                    timeout=120
                )

                # Check response
                response.raise_for_status()

                # Extract results
                result = response.json()

                predictions = pd.DataFrame(
                    result["predictions"]
                )

                # Display results
                st.subheader("Prediction Results")

                st.dataframe(
                    predictions,
                    use_container_width=True
                )

                # ------------------------------------------------
                # Allow user to download predictions
                # ------------------------------------------------

                csv_data = predictions.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions",
                    data=csv_data,
                    file_name="superkart_sales_predictions.csv",
                    mime="text/csv"
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Unable to connect to the backend API: {e}"
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )
