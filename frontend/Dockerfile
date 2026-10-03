
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all frontend files into the container
COPY . .

# Install dependencies from the requirements file
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
