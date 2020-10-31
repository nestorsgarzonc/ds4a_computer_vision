FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./app /app
EXPOSE 80
#COPY requirements.txt .
#RUN pip --no-cache-dir install -r requirements.txt
RUN pip install opencv-python
RUN pip install fastapi uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]