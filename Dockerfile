# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . "$HOME/.cargo/env" \
    && rustup default stable \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Copy only requirements files first to leverage Docker cache
COPY requirements.txt .
COPY ./minirag ./minirag
COPY setup.py .

# Install dependencies
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir jupyter notebook
RUN pip install --user --no-cache-dir transformers torch sentence-transformers
RUN pip install --user --no-cache-dir openai nano-vectordb

RUN pip install .

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/minirag ./minirag
COPY --from=builder /app/setup.py .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Jupyter Notebook用のポートを公開
EXPOSE 8165

# Jupyter Notebook用の設定ディレクトリを作成
RUN mkdir -p /root/.jupyter

# デフォルトでJupyter Notebookを起動
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8165", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]