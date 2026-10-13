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

project(tensorflow-ruy-populate NONE)



include(ExternalProject)
ExternalProject_Add(tensorflow-ruy-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip" "URL_MD5" "SOURCE_SUBDIR" "NON_EXISTENT_FOLDER"
                    SOURCE_DIR          "/home/ronren/embgemma2/fvp/dyn_u85/_deps/tensorflow-ruy-src"
                    BINARY_DIR          "/home/ronren/embgemma2/fvp/dyn_u85/_deps/tensorflow-ruy-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


