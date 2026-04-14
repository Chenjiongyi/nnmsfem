CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

FIND_PACKAGE(deal.II 9.5.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

SET(TARGET "nnmsfem")
SET(TARGET_SRC ${TARGET}.cc)

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

SET(ONNXRUNTIME_DIR "/home/dealii/onnx/onnxruntime-linux-x64-1.18.1")
TARGET_INCLUDE_DIRECTORIES(${TARGET} SYSTEM PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_directories(${TARGET} PRIVATE ${ONNXRUNTIME_DIR}/lib)
TARGET_LINK_LIBRARIES(${TARGET} onnxruntime)