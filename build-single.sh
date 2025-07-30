#!/usr/bin/env bash
# ===================================================================
# build-single.sh
# Build-time script that produces a *single static* native binary
#   ./build/release/nlproto-native
# with zero external jar dependencies after first run.
# ===================================================================

set -euo pipefail
shopt -s extglob

# --- parameterisable ------------------------------------------------
CLONE_ROOT="${CLONE_ROOT:-nlproto-native}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$(pwd)/nlproto-native/install}"
JAVA_HOME="${JAVA_HOME:-$(readlink -f /usr/lib/jvm/java-21-openjdk)}"

# --- helpers --------------------------------------------------------
info()  { printf '[INFO] %s\n' "$*" ; }
die()   { printf '[ERROR] %s\n' "$*" ; exit 1; }

# 1) clone repository -------------------------------------------------
if [[ ! -d "$CLONE_ROOT" ]]; then
    info "Cloning source repository …"
    git clone --depth 1 \
        https://github.com/YourOrg/nlproto-native.git "$CLONE_ROOT"
fi
cd "$CLONE_ROOT" || die "Failed to cd into $CLONE_ROOT"

# 2) Java build (netlogo jar + custom glue) --------------------------
if [[ ! -f "java/target/netlogo-core.jar" ]]; then
    info "Building Java glue once …"
    if command -v mvn >/dev/null 2>&1; then
        MAVEN_EXEC="mvn"
    elif command -v ./mvnw >/dev/null 2>&1 && [[ -x ./mvnw ]]; then
        MAVEN_EXEC="./mvnw"
    else
        die "Maven or mvnw not in PATH"
    fi

    # NetLogo 6.4 jars (one-shot download)
    "$MAVEN_EXEC" -f java/pom.xml --batch-mode package -DskipTests
fi

# 3) Native build ----------------------------------------------------
CMAKE_TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE="cmake/tooltoolchain.cmake"
if ! [[ -f "build/${BUILD_TYPE}/Makefile" ]]; then
    mkdir -p build/${BUILD_TYPE}
    cmake -B build/${BUILD_TYPE} \
          -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          -DJNI_INCLUDE_DIRS="${JAVA_HOME}/include;${JAVA_HOME}/include/linux" \
          -DJAVA_JVM_LIBRARY="${JAVA_HOME}/lib/server/libjvm.so" \
          -G Ninja
fi

info "Compiling native driver …"
cmake --build "build/${BUILD_TYPE}" --parallel$(nproc)

# 4) Strip & final outcome ------------------------------------------
strip --strip-all "build/${BUILD_TYPE}/nlproto-native" || true

info "Build finished -> $(pwd)/build/${BUILD_TYPE}/nlproto-native"
info "Run quick check:"
info "   ./build/${BUILD_TYPE}/nlproto-native --help"