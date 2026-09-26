FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && useradd --create-home evaluator \
    && mkdir /app/artifacts && chown evaluator:evaluator /app/artifacts
USER evaluator
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "evaluation_xai.api:app", "--host", "0.0.0.0", "--port", "8000"]
