FROM python:3.12-slim

# Evita arquivos .pyc, habilita logs imediatos e define o local da venv.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copia o executavel do uv da imagem oficial.
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

# Instala primeiro as dependencias para aproveitar o cache do Docker.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copia somente o necessario para executar a API.
COPY src/ ./src/
COPY models/logistic_regression_final.joblib ./models/logistic_regression_final.joblib

# A aplicacao nao precisa executar como root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
    
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
