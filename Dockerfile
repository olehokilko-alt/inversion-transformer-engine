# Inversion Transformer Enterprise - Secure Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies
# build-essential is needed for Cython compilation (gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
# Ensure Cython is installed for build process
RUN pip install --no-cache-dir Cython==3.0.0
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Source Code
COPY core/ ./core/
COPY train/ ./train/
COPY serve/ ./serve/
COPY setup_cython.py .

# 4. SECURE COMPILATION STEP
# This compiles .py files to .so (shared objects) and DELETES the original .py files.
# This ensures the container contains NO readable source code for the core logic.
RUN python setup_cython.py build_ext --inplace \
    && rm core/adaptive_controller.py \
    && rm core/inversion_transformer.py \
    && rm setup_cython.py \
    && rm -rf build

# 5. Environment Setup
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 6. Launch API Server
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
