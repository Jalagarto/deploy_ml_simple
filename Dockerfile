# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for faster builds
COPY app/model_pipeline.joblib /app/

# Copy the Python application code
COPY app/ml_api /app/ml_api

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set the entrypoint for the container to run the FastAPI app
CMD ["uvicorn", "ml_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

