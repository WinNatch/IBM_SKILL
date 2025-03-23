# ✅ Use a prebuilt PyTorch image with CUDA support for efficiency
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ✅ Set the working directory inside the container
WORKDIR /app

# ✅ Copy only requirements.txt first to take advantage of Docker layer caching
COPY requirements.txt .

# ✅ Install dependencies without cache to save space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Pre-download NLTK stopwords & wordnet (if needed) 
RUN mkdir -p /app/nltk_data && \
    python -c "import nltk; nltk.data.path.append('/app/nltk_data'); \
               nltk.download('stopwords', download_dir='/app/nltk_data'); \
               nltk.download('wordnet', download_dir='/app/nltk_data')"

# ✅ Set NLTK data path environment variable
ENV NLTK_DATA=/app/nltk_data

# ✅ Copy only necessary files (excluding large files)
COPY . .

# ✅ Set environment variable for Cloud Run
ENV PORT=8080

# ✅ Expose port
EXPOSE 8080

# ✅ Ensure Cloud Run recognizes Gunicorn as the main process
ENV PYTHONUNBUFFERED=TRUE

# ✅ Run the Flask app using Gunicorn for performance
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "--timeout", "120", "--log-level", "debug", "--access-logfile", "-", "app:app"]