FROM python:3.12

# Install system dependencies buat OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Expose port
EXPOSE 8080

# Jalankan app pakai gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:application"]
