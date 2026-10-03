
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all backend files into the container
# This includes app.py, requirements.txt,
# and superkart_model.joblib
COPY . .

# Install dependencies from the requirements file
# without using cache to reduce image size
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Start the Flask application using Gunicorn
#
# -w 4:
#   Uses 4 worker processes to handle requests
#
# -b 0.0.0.0:7860:
#   Makes the API accessible on all network interfaces
#   through port 7860
#
# app:superkart_sales_api:
#   Loads the Flask application object named
#   'superkart_sales_api' from app.py

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_sales_api"]
