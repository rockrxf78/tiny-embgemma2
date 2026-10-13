# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-src"
  "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-build"
  "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix"
  "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/tmp"
  "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp"
  "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src"
  "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ronren/embgemma2/fvp/dyn_u55/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
