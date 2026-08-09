# Use a lightweight Python 3.14 base image.
FROM python:3.14-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the dependency definition separately to take
# advantage of Docker layer caching.
COPY requirements.txt .

# Install the Python dependencies required by CLARA.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the CLARA source code and FastAPI application.
COPY src ./src
COPY app ./app

# Document the port exposed by the FastAPI service.
EXPOSE 8000

# Start the CLARA API using Uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]