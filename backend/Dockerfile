# Use an official lightweight Python image
FROM python:3.11-slim

# Install system dependencies (needed for OCR and PDF processing)
RUN apt-get update && apt-get install -y \
  libgl1 \
  poppler-utils \
  tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user — HF Spaces enforces UID 1000
RUN useradd -m -u 1000 user

# Copy and install requirements as root (global install)
COPY req.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt

# Switch to non-root user BEFORE copying app code
USER user

# Set environment for the non-root user
ENV HOME=/home/user \
  PATH="/home/user/.local/bin:${PATH}" \
  HF_HOME=/home/user/.cache/huggingface \
  TRANSFORMERS_CACHE=/home/user/.cache/huggingface/hub

WORKDIR /home/user/app

# Copy application code with correct ownership
COPY --chown=user:user . .

# Verify that critical files are REAL binaries (not LFS pointers)
# LFS files should be >1KB; pointers are <200 bytes
RUN echo "=== VERIFYING FILE SIZES (LFS Check) ===" && \
  for f in faiss_store/index.faiss faiss_store/index.pkl website_data.json; do \
  if [ -f "$f" ]; then \
  size=$(stat --printf="%s" "$f"); \
  echo "$f → ${size} bytes"; \
  if [ "$size" -lt 500 ]; then \
  echo "❌ FATAL: $f is still an LFS pointer!"; \
  head -c 200 "$f"; \
  echo ""; \
  echo "The GitHub Actions workflow must push real files, not pointers."; \
  exit 1; \
  fi; \
  else \
  echo "⚠️  $f not found (may be expected if not yet generated)"; \
  fi; \
  done && \
  echo "✅ All critical files verified as real binaries."

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Command to run the application on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]