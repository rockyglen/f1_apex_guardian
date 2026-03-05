FROM python:3.12-slim

WORKDIR /app

# Install Linux system dependencies for FastF1 and Plotly
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Hugging Face uses port 7860 by default for Docker Spaces
EXPOSE 7860

# Start Streamlit on the correct port
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]