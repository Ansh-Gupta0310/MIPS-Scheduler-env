# Dockerfile for MIPS Scheduler Environment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment package files into the container
COPY . /app/

# Install openenv-core first from PyPI, then the environment package
RUN pip install --no-cache-dir "openenv-core[core]>=0.2.2" && \
    pip install --no-cache-dir .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true
ENV MIPS_SCHEDULER_TASK=easy_alu_chain

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "mips_scheduler_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
