Yaml files contains the 4 .yaml files which were used for creating persistent storage, running train job, running the inference deployment and exposing the service.
(pvc.yaml, train.yaml, infer.yaml and service.yaml)

The training folder conatins the Dockerfile and training files.
To build run-
cd training
docker build -t mnisttrain .

The infernce folder conatins the code for the infernce flask app and also the dockerfile