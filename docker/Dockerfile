FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/work/src \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /work

# The repository is mounted at /work rather than copied in, so that a code
# change does not rebuild the image. The image carries the environment only.
CMD ["python", "-c", "import numpy, scipy, biotite, yaml; print('span environment ready')"]
