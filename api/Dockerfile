FROM python:3.7-alpine
LABEL maintainer="Anthony Dickson <eight0153@gmail.com>"

COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY src/ /app
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["app.py"]