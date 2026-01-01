# Dockerfile for compiling Core Logic on Linux
FROM python:3.10-slim AS builder

WORKDIR /build

# Install compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Cython
RUN pip install --no-cache-dir Cython==3.0.0

# Copy Source
COPY setup_cython.py .
COPY core/ ./core/

# Compile
RUN python setup_cython.py build_ext --inplace

# Verify compilation
RUN ls -la core/*.so

# This container is just a "factory". We will copy files out of it.
CMD ["echo", "Compilation Complete"]
