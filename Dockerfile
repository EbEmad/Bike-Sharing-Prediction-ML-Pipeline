# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app


# Copy the requirements file into the container

COPY requirements.txt .

# Install Python dependencies
RUN pip install  -r requirements.txt

# Copy the rest of the application code into the container
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Set the entrypoint to run your shell script
CMD ["bash", "/run.sh"]