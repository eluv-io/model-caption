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

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

COPY setup.py .
RUN mkdir caption

RUN /opt/conda/envs/caption/bin/pip install .

COPY models ./models

COPY config.yml run.py config.py .
COPY caption ./caption

ENTRYPOINT ["/opt/conda/envs/caption/bin/python", "run.py"]FROM continuumio/miniconda3:latest
