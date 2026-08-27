FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-deps \
    torch==2.13.0+cpu \
    torchvision==0.28.0+cpu

# PyTorch runtime dependencies
RUN pip install --no-cache-dir \
    sympy \
    networkx \
    filelock \
    typing-extensions \
    jinja2 \
    fsspec

COPY app/ ./app/
COPY src/ ./src/
COPY splits_vocab/ ./splits_vocab/
COPY weights/ ./weights/
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 7860

CMD ["./start.sh"]