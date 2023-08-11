FROM python:3.8-buster as runtime

LABEL maintainer="Sotirios Moschos <sotiriosmos@gmail.com>"

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    ffmpeg libsm6 libxext6  -y

ENV PATH="/opt/conda/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN useradd -m satori
WORKDIR /home/satori

COPY dev dev

RUN chmod +x run.sh && chown -R satori /home/satori

USER satori
CMD ["python", "-c", "print('Hello from the development container!')"]
