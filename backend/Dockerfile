FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files and folders
COPY models/ ./models/
COPY app.py ./
COPY datacube_generator.py ./
COPY image_processor.py ./
COPY config.py ./
COPY dependencies.py ./
COPY image_drawer.py ./

EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]