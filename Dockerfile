# Start from the PyTorch Lightning base image with CUDA support
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.8.0

# Install system dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir transformers \
    pip install --no-cache-dir accelerate \
    pip install --no-cache-dir bitsandbytes \
    pip install --no-cache-dir packaging \
    pip install --no-cache-dir termcolor \
    pip install --no-cache-dir ujson \
    pip install --no-cache-dir tqdm \
    pip install --no-cache-dir google-cloud-documentai \
    pip install --no-cache-dir ninja \
    pip install --no-cache-dir langchain

# Add custom local CUDA installation to the image
# Required to install flash-attn
ADD cuda/ /usr/local/cuda/

# Install flash-attn with specific build settings
# --no-build-isolation ensures it uses the existing environment for building
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Clean up: remove the added CUDA directory to reduce image size
# Note: This doesn't reduce the image size of previous layers
RUN rm -rf /usr/local/cuda/