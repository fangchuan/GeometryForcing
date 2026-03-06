# FROM nvcr.io/nvidia/pytorch:24.05-py3
FROM registry.qunhequnhe.com/mri/nvidia/pytorch:24.05-py3

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies if needed (e.g., for OpenCV or other media libraries usually required in video processing)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies directly into the container's optimized Python environment
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y && pip install opencv-python-headless==4.7.0.72 -q

# Copy the rest of the project files
COPY . .

# Set the default command to bash
CMD ["/bin/bash"]
