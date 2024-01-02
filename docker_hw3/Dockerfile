#Use the Pytorch image
FROM pytorch/pytorch

# Copy the all source code into the container.
COPY . /app
WORKDIR /app

USER root

# Run the application.
CMD python mnist.py --batch-size 32 --test-batch-size 32 --epochs 1 --no-cuda --no-mps