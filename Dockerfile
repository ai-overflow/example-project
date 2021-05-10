FROM tiangolo/uwsgi-nginx-flask:python3.8-alpine
# RUN apk --update add bash nano
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev libffi-dev libjpeg-turbo-dev zlib-dev
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static

# Don't put these in requirements.txt, 
# tritonclient is a compile package and 
# takes some time to compile...
RUN pip install nvidia-pyindex
RUN pip install tritonclient[all]

COPY ./src/requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt