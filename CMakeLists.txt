cmake_minimum_required(VERSION 3.18)
project(nlserver)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Threads & POSIX deps --------------------------------------------------
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(JNI REQUIRED)   # JAVA headers

# 2. External deps ---------------------------------------------------------
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(CLI11 REQUIRED)
find_package(PkgConfig REQUIRED) # for benchmark (optional)
pkg_check_modules(benchmark REQUIRED IMPORTED_TARGET benchmark)
# extras
find_package(PkgConfig)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)

# 3. PROTO COLLECTION   ———————————————————————————————————————————————
execute_process(
    COMMAND
      mvn -q -f "${CMAKE_CURRENT_SOURCE_DIR}/pom.xml" install:install-file
             -Dfile="/Applications/NetLogo 6.4.0/app/netlogo-6.4.0.jar"  # NetLogo jar location from config
             -DgroupId=org.nlogo
             -DartifactId=netlogo
             -Dversion=6.4.0
             -Dpackaging=jar
             -DgeneratePom=true
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE MAVEN_RESULT
)
if(NOT MAVEN_RESULT EQUAL 0)
  message(WARNING "Maven failed – build will continue but Java components may not be usable")
endif()

# 4. C++ util library -------------------------------------------------------
add_library(nlutil STATIC
    src/async_writer.cpp
    src/benchmark1.cpp
    src/config.cpp
    src/expander.cpp
    src/jni_bridge.cpp
    src/jvm_manager.cpp
    src/retry_policy.cpp
    src/results_ring.cpp
    src/thread_pool_worker.cpp
    src/output_manager.cpp
    src/config_parser.cpp
)
target_include_directories(nlutil
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC  ${JNI_INCLUDE_DIRS}
)
target_link_libraries(nlutil
    PUBLIC  ${JNI_LIBRARIES}              # JNI headers & lib
    PUBLIC  Threads::Threads
    PUBLIC  PkgConfig::benchmark
    PUBLIC  yaml-cpp::yaml-cpp
    PUBLIC  spdlog::spdlog
    PUBLIC  PkgConfig::ZSTD
)

# 5. Command-line binary ----------------------------------------------------
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME} 
    PRIVATE nlutil
    PRIVATE CLI11::CLI11
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)