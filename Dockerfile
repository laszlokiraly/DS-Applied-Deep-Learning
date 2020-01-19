FROM python:3.7.6-buster

EXPOSE 8080

ADD ./transfer /transfer
ADD ./hrnet-imagenet-valid /hrnet-imagenet-valid

COPY ./requirements.txt /

RUN pip install -r requirements.txt

CMD python transfer/server.py