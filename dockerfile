FROM python:3.10-slim

#this will become of our "root" inside the docker
WORKDIR /code

COPY requirements.txt requirements.txt

#install whatever requirements we need
RUN pip install --no-cache-dir --upgrade -r requirements.txt    

