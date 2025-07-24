FROM public.ecr.aws/lambda/python:3.10

# Set the working directory inside the container
WORKDIR /var/task

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files and folders
COPY models/ ./models/
COPY app.py ./
COPY datacube_generator.py ./
COPY image_processor.py ./

# Command to run the application
CMD ["app.handler"]