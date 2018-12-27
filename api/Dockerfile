FROM python:3.7-alpine
LABEL maintainer="Anthony Dickson <eight0153@gmail.com>"

COPY requirements.txt /
RUN apk update \
    && apk add libpq \
    && apk add --no-cache --virtual .build-deps postgresql-dev gcc musl-dev \
    && pip install --no-cache-dir -r /requirements.txt \
    && apk del --no-cache .build-deps
COPY . /app
WORKDIR /app

ENTRYPOINT ["sh", "entrypoint.sh"]
