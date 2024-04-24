# Pull base image
# See: https://github.com/SeleniumHQ/selenium/issues/13091
FROM selenium/standalone-chrome:4.15.0-20231122

# Switch to root user for Linux installations
USER root

# Configure Python logs to stream to stdout without buffering
ENV PYTHONUNBUFFERED 1

# Install dependencies for Debian distro
RUN apt update -y && apt upgrade -y && \
    apt install -y --no-install-recommends \
        build-essential \
        libatlas-base-dev \
        libgdal-dev \
        gfortran \
        python3 \
        python3-pip

# Set working directory in container
WORKDIR /perpetual/pipeline

# Install requirements based on declared environment
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=100

# Copy remaining application code to container
COPY . .

# Run the pipeline
CMD ["bash", "setup.sh", "--migrate"]
