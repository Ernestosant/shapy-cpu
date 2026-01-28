# ==============================================================================
# SHAPY CPU Docker Image
# ==============================================================================
# Multi-stage build for body shape estimation from single RGB images.
# Runs entirely on CPU (no GPU required).
#
# Build: docker build -t shapy-cpu .
# Run:   docker run -v ./input:/app/input -v ./output:/app/output -v ./SHAPY_FILES:/app/data shapy-cpu --image /app/input/photo.jpg
# ==============================================================================

FROM python:3.8-slim-bullseye AS base

# System dependencies for pyrender (OSMesa), OpenCV, image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    cmake \
    # OpenGL / OSMesa for CPU rendering
    libosmesa6 \
    libosmesa6-dev \
    freeglut3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    # Image processing - libjpeg-turbo for jpeg4py
    libjpeg-dev \
    libpng-dev \
    libturbojpeg0 \
    libturbojpeg0-dev \
    # OpenCV dependencies
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Fonts for visualization
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CPU rendering
ENV PYOPENGL_PLATFORM=osmesa
ENV MPLBACKEND=Agg

WORKDIR /app

# ==============================================================================
# Python Dependencies Stage
# ==============================================================================

# Install PyTorch CPU version first (large download)
RUN pip install --no-cache-dir \
    torch==1.7.1+cpu \
    torchvision==0.8.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --ignore-installed torch torchvision

# Install additional dependencies
RUN pip install --no-cache-dir \
    smplx==0.1.28 \
    trimesh==3.9.1 \
    pyrender==0.1.45

# Install MediaPipe with compatible OpenCV
# MediaPipe requires opencv-contrib-python, so uninstall opencv-python first
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-contrib-python==4.8.1.78 mediapipe==0.10.8

# ==============================================================================
# Application Stage
# ==============================================================================

# Install attributes module (without mesh-mesh-intersection which requires CUDA)
COPY attributes/ attributes/
RUN cd attributes && pip install --no-cache-dir -e .

# Copy source code
COPY regressor/ regressor/
COPY measurements/ measurements/
COPY samples/ samples/
COPY body_measurements/ body_measurements/
COPY infer.py .
COPY visualize.py .
COPY detect_keypoints.py .

# Create directories for mounted volumes
RUN mkdir -p /app/input /app/output /app/data

# ==============================================================================
# Runtime Configuration
# ==============================================================================

# Volumes for data and input/output
VOLUME ["/app/data", "/app/input", "/app/output"]

# Set Python path - include body_measurements for CPU version
ENV PYTHONPATH=/app:/app/regressor:/app/attributes:/app/measurements:/app/body_measurements

# Default number of threads (can be overridden)
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import smplx; print('OK')" || exit 1

# Default entrypoint
ENTRYPOINT ["python", "infer.py"]

# Default command (can be overridden)
CMD ["--help"]

# ==============================================================================
# Labels
# ==============================================================================
LABEL maintainer="SHAPY CPU"
LABEL description="SHAPY body shape estimation for CPU inference"
LABEL version="1.0"
