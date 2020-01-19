FROM adl-pytorch-base:latest

EXPOSE 8080

ADD ./transfer /transfer
ADD ./hrnet-imagenet-valid /hrnet-imagenet-valid

CMD python transfer/server.py
