FROM python:3.9-slim

#this will become of our "root" inside the docker
WORKDIR /code

COPY requirements.txt requirements.txt

#install whatever requirements we need
RUN pip install --no-cache-dir --upgrade -r requirements.txt    

#local working dir to docker working dir (inside /code)
COPY . .

#cmd[execute, everything else are arguments]; 0.0.0.0 to listen on all ports
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]