FROM continuumio/miniconda3:latest

WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential \
    && apt-get install -y ffmpeg

RUN \
   conda create -n caption python=3.8 -y

SHELL ["conda", "run", "-n", "caption", "/bin/bash", "-c"]

RUN \
    conda install -y cudatoolkit=10.1 cudnn=7 nccl && \
    conda install -y -c conda-forge ffmpeg-python

COPY caption ./caption
COPY config.yml run.py setup.py config.py .

RUN /opt/conda/envs/caption/bin/pip install .

ENTRYPOINT ["/opt/conda/envs/caption/bin/python", "run.py"]