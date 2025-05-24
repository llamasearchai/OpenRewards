FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.cargo/bin:$PATH"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libpq-dev \
    pkg-config \
    nodejs \
    npm \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN rustup default stable && rustup component add rustfmt clippy

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml poetry.lock* ./

# Install Poetry
RUN pip3 install poetry
RUN poetry config virtualenvs.create false

# Install Python dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy Rust code and build components
COPY rust ./rust/
COPY Cargo.toml Cargo.lock* ./
RUN cargo build --release

# Copy Python code
COPY python ./python/

# Install the Rust components with maturin
RUN pip3 install maturin
RUN maturin develop --release

# Copy Tauri application
COPY tauri ./tauri/

# Build Tauri app
WORKDIR /app/tauri
RUN npm install
RUN npm run tauri build

# Final stage with minimal runtime
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Copy artifacts from build stage
COPY --from=base /app/tauri/src-tauri/target/release/reward-modeling-dashboard /usr/local/bin/
COPY --from=base /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy Python code (runtime only)
COPY python ./python/

# Expose ports
EXPOSE 3000
EXPOSE 8000

# Start application
CMD ["/usr/local/bin/reward-modeling-dashboard"]