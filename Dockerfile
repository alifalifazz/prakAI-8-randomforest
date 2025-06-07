# Pakai base image Python slim
FROM python:3.12-slim

# Install libgl1-mesa-glx supaya OpenCV bisa jalan
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Set working directory di container
WORKDIR /app

# Copy file requirements.txt dan install dependencies Python
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project ke container
COPY . .

# Expose port yang Railway gunakan (biasanya 8080)
EXPOSE 8080

# Perintah menjalankan app, pastikan pointing ke "application"
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:application"]
