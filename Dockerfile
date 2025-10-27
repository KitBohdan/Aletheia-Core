FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md requirements-dev.txt ./
RUN pip install -U pip && pip install -r requirements-dev.txt && pip install .
COPY vct ./vct
EXPOSE 8000
ENV VCT_SIMULATE=1
CMD ["uvicorn", "vct.api.app:app", "--host", "0.0.0.0", "--port", "8000"]