#FROM python:3.7
#RUN pip install fastapi uvicorn
#EXPOSE 8080
#EXPOSE 80
#EXPOSE 15400
#CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80" ]

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
ADD ./app /app
COPY ./app /app
COPY . ./
COPY ./app ./

RUN pip install opencv-python
RUN pip install aiofiles
RUN pip install python-multipart
RUN pip install opencv-contrib-python
COPY ./app /app/app