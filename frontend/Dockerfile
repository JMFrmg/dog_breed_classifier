FROM python:3.9-slim

COPY ./app /code
COPY requirements.txt /code

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*



RUN pip3 install -r requirements.txt

#EXPOSE 8000

#HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8000", "--server.address=0.0.0.0"]