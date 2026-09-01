# ============================================================================
# Dockerfile para Text-Simplification-ISC
#
# Requiere Python >= 3.13 (ver pyproject.toml). El runtime python nativo de
# Render no llega a 3.13, por eso usamos una imagen Docker explícita.
#
# Todo el sistema (Streamlit + 2 servidores MCP) se ejecuta en UN solo
# contenedor. El CMD lanza start_prod.py, un supervisor que arranca los dos
# servidores MCP en 127.0.0.1 (internos, no expuestos) y luego Streamlit en
# primer plano. Así cabe en un único web service gratuito de Render.
# ============================================================================
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instalar dependencias del sistema ligero (opcional; algunas ruedas lo necesitan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Primero las dependencias para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Después el código de la aplicación
COPY . .

# Streamlit headless y desactivar telemetría
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MCP_METRICS_HOST=127.0.0.1 \
    MCP_METRICS_PORT=8020 \
    MCP_SEARCH_HOST=127.0.0.1 \
    MCP_SEARCH_PORT=8021 \
    MCP_METRICS_SERVER_URL=http://127.0.0.1:8020/mcp/ \
    MCP_SEARCH_SERVER_URL=http://127.0.0.1:8021/mcp/

EXPOSE 8501

CMD ["python", "start_prod.py"]
