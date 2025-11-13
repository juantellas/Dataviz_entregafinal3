# --- Imagen base ---
FROM python:3.11-slim

# --- Evitar buffer en stdout/stderr ---
ENV PYTHONUNBUFFERED=1

# --- Establecer directorio de trabajo ---
WORKDIR /app

# --- Copiar requirements y luego instalar (cacheable) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copiar toda la app ---
COPY . .

# --- Exponer puerto ---
EXPOSE 8050

# --- Comando por defecto ---
CMD ["python", "app.py"]
