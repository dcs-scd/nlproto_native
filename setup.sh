#!/bin/bash
# setup.sh - NetLogo Simulation System Setup

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    NETLOGO_PATH="/Applications/NetLogo 6.4.0"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"  
    NETLOGO_PATH="/opt/netlogo"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Install dependencies
if [[ "$OS" == "macos" ]]; then
    brew install cmake maven yaml-cpp spdlog cli11 google-benchmark zstd pkg-config openjdk
elif [[ "$OS" == "linux" ]]; then
    sudo apt update
    sudo apt install -y cmake maven build-essential openjdk-11-jdk pkg-config \
                        libyaml-cpp-dev libspdlog-dev libcli11-dev libbenchmark-dev libzstd-dev
fi

# Update CMakeLists.txt
sed -i "s|/Applications/NetLogo 6.4.0|$NETLOGO_PATH|g" CMakeLists.txt

# Update example files
find examples/ -name "*.yaml" -exec sed -i "s|/Applications/NetLogo 6.4.0|$NETLOGO_PATH|g" {} \;

# Build system
mkdir -p build && cd build
cmake ..
make -j$(nproc)

echo "✅ Setup complete! Test with: ./build/nlserver examples/01_beginner_basic_sweep.yaml"
