FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ARG USERNAME=vscode
ARG UID=1000
ARG GID=${UID}

COPY library-scripts/*.sh library-scripts/*.env /tmp/library-scripts/
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && /bin/bash /tmp/library-scripts/common-debian.sh \
    && apt-get install -y python3 python3-pip && pip install --upgrade pip --no-input \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

RUN su - vscode -c "curl -sSL https://install.python-poetry.org | python3 -"
