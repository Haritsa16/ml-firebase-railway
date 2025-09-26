# Gunakan Python image
FROM python:3.10

# Set working dir
WORKDIR /app

# Copy semua file
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan Flask
CMD ["python", "app.py"]
