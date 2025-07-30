FROM ubuntu:22.04
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y openjdk-17-jdk-headless g++ cmake pkg-config
COPY . /workspace
WORKDIR /workspace
WORKDIR /workspace
CMD ["./run-headless.sh"]