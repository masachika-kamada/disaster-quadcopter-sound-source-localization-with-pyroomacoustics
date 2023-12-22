FROM mcr.microsoft.com/devcontainers/python:3.10

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ffmpeg \
    liblapack-dev \
    liblapacke-dev \
    gcc \
    g++

ENV TZ Asia/Tokyo

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git clone https://github.com/bnaecker/pygsvd.git && \
    cd pygsvd && \
    python setup.py build_ext --inplace && \
    pip install .
