FROM nvcr.io/nvidia/pytorch:22.02-py3 

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

RUN apt-get update && apt-get install -y \
    sudo \
    python3 git-core zip unzip tmux htop \
  && rm -rf /var/lib/apt/lists/*

ARG WORKSPACE_DIR="/ws"
ENV HOME_PATH="${WORKSPACE_DIR}/home"

RUN mkdir -p "${WORKSPACE_DIR}"

RUN groupadd --gid 1000 user \
  && useradd --uid 1000 --gid user --shell /bin/bash --create-home user --home-dir "${HOME_PATH}" \
  && mkdir -p /etc/sudoers.d \
  && echo 'user ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers.d/user \
  && chmod 440 /etc/sudoers.d/user

COPY ./docker-entrypoint.sh /bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

WORKDIR "$WORKSPACE_DIR"
CMD ["bash"]

