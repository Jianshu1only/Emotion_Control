# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12

# Set the working directory in the container
# WORKDIR /app/primary_emotions
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt --use-deprecated=legacy-resolver

# Make port 80 available to the world outside this container
EXPOSE 8501

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
CMD ["bash", "-c","streamlit run interactive_emotion_generator.py"]