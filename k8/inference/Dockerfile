# Using pytorch's base image (latest version by default)
FROM python:3.9-slim

# Copying the application files to container
COPY . /app


# Setting working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Root privileges granted
USER root

# Command to run once container starts - arguments passed
# no-cuda - to disable cuda training,
# no-mps - to disable MacOS gpu training
# epochs and batchsize
EXPOSE 5000
# ENV FLASK_APP=app.py
CMD ["flask", "run", "--host","0.0.0.0", "--port","5000" ]
