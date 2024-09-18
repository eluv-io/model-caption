FROM continuumio/miniconda3:latest

WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential 

RUN \
   conda create -n caption python=3.8 -y

SHELL ["conda", "run", "-n", "caption", "/bin/bash", "-c"]

RUN \
    conda install -y cudatoolkit=10.1 cudnn=7 nccl

COPY . .

RUN /opt/conda/envs/caption/bin/pip install .

ENTRYPOINT ["/opt/conda/envs/asr/bin/python", "run.py"]