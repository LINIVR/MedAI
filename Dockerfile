# 1. Use Python 3.10 base image
FROM python:3.10-slim

# 2. Set clean environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set working directory inside container
WORKDIR /app

# 4. Install system tools for OCR, model files, etc.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy full project (including streamlit_app.py, models, PDFs, etc.)
COPY . .

# 7. Expose default Streamlit port for Hugging Face
EXPOSE 7860

# 8. Run the app (updated to run streamlit_app.py from root)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.enableCORS=false"]
