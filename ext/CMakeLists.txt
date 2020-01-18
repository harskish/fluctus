include(CheckCXXCompilerFlag)

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/nanogui/include")
  message(FATAL_ERROR "Fluctus dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by calling \"git submodule update --init --recursive\"")
endif()

# Set a default build configuration (Release)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

# Enable folders for projects in Visual Studio
if (CMAKE_GENERATOR MATCHES "Visual Studio")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

  if (HAS_CPP14_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif (HAS_CPP11_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "Unsupported compiler -- Fluctus requires C++11 support!")
  endif()

  # Enable link time optimization and set the default symbol
  # visibility to hidden (very important to obtain small binaries)
  if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    # Default symbol visibility
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  endif()
endif()

# Sanitize build environment for static build with C++11
if (MSVC)
  # Disable annoying secure CRT warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")

  # We'll select the TBB library ourselves
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__TBB_NO_IMPLICIT_LINKAGE")

  # Parallel build on MSVC (all targets)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

  if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")

    # Disable Eigen vectorization for Windows 32 bit builds (issues with unaligned access segfaults)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEIGEN_DONT_ALIGN")
  endif()

  # Static build
  set(CompilerFlags
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
  foreach(CompilerFlag ${CompilerFlags})
    string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
  endforeach()
elseif(APPLE)
  # Try to auto-detect a suitable SDK
  execute_process(COMMAND bash -c "xcodebuild -version -sdk | grep MacOSX | grep Path | head -n 1 | cut -f 2 -d ' '" OUTPUT_VARIABLE CMAKE_OSX_SYSROOT)
  string(REGEX REPLACE "(\r?\n)+$" "" CMAKE_OSX_SYSROOT "${CMAKE_OSX_SYSROOT}")
  string(REGEX REPLACE "^.*X([0-9.]*).sdk$" "\\1" CMAKE_OSX_DEPLOYMENT_TARGET "${CMAKE_OSX_SYSROOT}")
endif()

# Build NanoGUI
set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_SHARED  OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
set(NANOGUI_USE_GLAD      ON CACHE BOOL " " FORCE)
add_subdirectory(nanogui)
set_property(TARGET glfw glfw_objects nanogui nanogui-obj PROPERTY FOLDER "dependencies")

# Setup custom search paths on Windows
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
	list(APPEND CMAKE_PREFIX_PATH
	 "${CMAKE_CURRENT_SOURCE_DIR}/windows/DevIL/x64")
	list(APPEND CMAKE_LIBRARY_PATH
	 "${CMAKE_CURRENT_SOURCE_DIR}/windows/DevIL/x64")
elseif("${CMAKE_GENERATOR}" MATCHES "Visual Studio" AND CMAKE_SIZEOF_VOID_P EQUAL 4)
	list(APPEND CMAKE_PREFIX_PATH
	 "${CMAKE_CURRENT_SOURCE_DIR}/windows/DevIL/Win32")
	list(APPEND CMAKE_LIBRARY_PATH
	 "${CMAKE_CURRENT_SOURCE_DIR}/windows/DevIL/Win32")
endif()

# Other dependencies
find_package(OpenCL 1.2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(DevIL REQUIRED)

# Pbrt-parser
add_subdirectory(pbrt-parser)
set(PBRT_PARSER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pbrt-parser/pbrtParser/include PARENT_SCOPE)

# CLT
add_subdirectory(CLT)
set(CLT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CLT/include PARENT_SCOPE)

# Fix broken DevIL include
if(IL_INCLUDE_DIR MATCHES ".*\/IL$")
	string(LENGTH "${IL_INCLUDE_DIR}" IL_INC_PATH_LEN)
	MATH(EXPR IL_INC_PATH_LEN "${IL_INC_PATH_LEN}-3")
	string(SUBSTRING "${IL_INCLUDE_DIR}" 0 "${IL_INC_PATH_LEN}" FIXED_IL_PATH)
	set(IL_INCLUDE_DIR "${FIXED_IL_PATH}" CACHE STRING "" FORCE)
endif()


set(GLFW_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/ext/glfw/include)
set(NANOVG_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/ext/nanovg/src)
set(NANOGUI_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/include)
set(EIGEN_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/nanogui/ext/eigen)


set(CompilerFlags
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELWITHDEBINFO COMPILE_DEFINITIONS U_CMAKE_BUILD_TYPE
      CMAKE_MACOSX_RPATH
      GLFW_INCLUDE_DIR
      NANOVG_INCLUDE_DIR NANOGUI_EXTRA_INCS NANOGUI_EXTRA_DEFS
	  NANOGUI_EXTRA_LIBS NANOGUI_INCLUDE_DIR EIGEN_INCLUDE_DIR
)
foreach(CompilerFlag ${CompilerFlags})
  set(${CompilerFlag} "${${CompilerFlag}}" PARENT_SCOPE)
endforeach()
