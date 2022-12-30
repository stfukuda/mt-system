ARG VARIANT=3.9

FROM python:${VARIANT}

RUN apt update && \
    apt upgrade -y && \
    apt install gnupg2 locales -y && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen && \
    pip install -U pip setuptools wheel poetry

ENV LANG=en_US.UTF-8
