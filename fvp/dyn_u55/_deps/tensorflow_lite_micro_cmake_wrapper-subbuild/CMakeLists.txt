# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.28.3)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(tensorflow_lite_micro_cmake_wrapper-populate NONE)



include(ExternalProject)
ExternalProject_Add(tensorflow_lite_micro_cmake_wrapper-populate
                     "UPDATE_DISCONNECTED" "False" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-software/-/raw/26.05/tflite_micro.cmake?ref_type=tags&inline=false" "URL_HASH" "MD5=52985fb8910dbaf034a17f9c6140b9d2" "DOWNLOAD_DIR" "/home/ronren/embgemma2/fvp/dyn_u55" "DOWNLOAD_NAME" "tflite_micro.cmake" DOWNLOAD_NO_EXTRACT YES
                    SOURCE_DIR          "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow_lite_micro_cmake_wrapper-src"
                    BINARY_DIR          "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow_lite_micro_cmake_wrapper-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


ExternalProject_Get_Property(tensorflow_lite_micro_cmake_wrapper-populate DOWNLOADED_FILE)
get_filename_component(dlFileName "${DOWNLOADED_FILE}" NAME)

ExternalProject_Add_Step(tensorflow_lite_micro_cmake_wrapper-populate copyfile
  COMMAND    "/usr/bin/cmake" -E copy_if_different
             "<DOWNLOADED_FILE>" "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow_lite_micro_cmake_wrapper-src"
  DEPENDEES  patch
  DEPENDERS  configure
  BYPRODUCTS "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow_lite_micro_cmake_wrapper-src/${dlFileName}"
  COMMENT    "Copying file to SOURCE_DIR"
)

