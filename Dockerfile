# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install dependencies for OpenCV, OpenGL, and GLib
RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    tcl-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run buzzwatch_app.py when the container launches
CMD ["python3", "./buzzwatch_analysis_app.py"]