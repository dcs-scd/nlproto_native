#!/usr/bin/env bash
# works on bash≥4 / zsh (Linux/macOS)

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"
./build-single.sh             # 1) native

java -cp .:src                 \
     -Djava.awt.headless=true  \
     netlogo.nativepipe.RunAllTests \
     "${1:-models/sample.nlogo}"