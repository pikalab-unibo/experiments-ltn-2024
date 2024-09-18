# Use an official PyTorch base image
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy the local files to the Docker container
COPY . /app

# Set an environment variable for the output directory
ENV OUTPUT_DIR=/app/output

# Create the output directory
RUN mkdir -p $OUTPUT_DIR

# Declare a volume for the output directory
VOLUME ["/app/output"]

# Install Python dependencies, including specific versions of torch and tensorboard
RUN pip install --no-cache-dir \
    scikit-learn \
    ucimlrepo \
    scipy \
    matplotlib \
    pandas \
    pyyaml \
    nbconvert==7.16.4 \
    nltk==3.8.1 \
    torch==2.3.1 \
    tensorboard==2.17.0

# Install ekuru_ltn from Test PyPI
RUN pip install -i https://test.pypi.org/simple/ ekuru_ltn

# Command to run the Python script
CMD ["python", "train_script.py"]