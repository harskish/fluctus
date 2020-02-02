cmake_minimum_required(VERSION 3.3)
project(Fluctus)

# OCL 1.2+ needed for clGetKernelArgInfo
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DNOMINMAX)

add_subdirectory(ext ext_build)

set(INCLUDE_DIRS
    ext/nanogui/include
    ${CLT_INCLUDE_DIR}
    ${PBRT_PARSER_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${NANOGUI_EXTRA_INCS}
    ${IL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    include
    src
)

set(LIBRARIES
    nanogui
    CLT
    pbrtParser
    ${NANOGUI_EXTRA_LIBS}
    ${OPENGL_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${IL_LIBRARIES}
    ${ILU_LIBRARIES}
)

set(SOURCE_FILES
    src/main.cpp
    src/clcontext.cpp
    src/clcontext.hpp
    src/window.cpp
    src/window.hpp
    src/progressview.cpp
    src/progressview.hpp
    src/geom.h
    src/bxdf_types.h
    src/tracer.cpp
    src/tracer_ui.cpp
    src/tracer.hpp
    src/bvh.hpp
    src/bvh.cpp
    src/sbvh.hpp
    src/sbvh.cpp
    src/bvhnode.hpp
    src/bvhnode.cpp
    src/rtutil.hpp
    src/triangle.hpp
    src/scene.cpp
    src/scene.hpp
    src/tinyfiledialogs.c
    src/tinyfiledialogs.h
    src/kernel_impl.hpp
    src/envmap.hpp
    src/envmap.cpp
    src/rgbe/rgbe.hpp
    src/rgbe/rgbe.cpp
    src/xxhash/xxhash.h
    src/xxhash/xxhash.c
    src/settings.cpp
    src/settings.hpp
    src/texture.cpp
    src/texture.hpp
    src/GLProgram.cpp
    src/GLProgram.hpp
    src/utils.h
    src/utils.cpp)

# Add configuration file if available
if (EXISTS "${CMAKE_SOURCE_DIR}/settings.json")
    file(GLOB PROJECT_CONFIGS settings.json)
endif()

# OptiX for AI denoiser, optional
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(OptiX 7)
if(NOT OptiX_INCLUDE)
    message(WARNING "OptiX 7 not found, disabling AI denoiser.")
else()
    message(STATUS "OptiX 7 found, building with AI denoiser.")
    find_package(CUDA REQUIRED)
    
    add_definitions(-DWITH_OPTIX) 
    set(INCLUDE_DIRS
        ${INCLUDE_DIRS}
        ${OptiX_INCLUDE}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
    )
    set(SOURCE_FILES
        ${SOURCE_FILES}
        src/denoiser/OptixDenoiser.hpp
        src/denoiser/OptixDenoiser.cpp
        src/denoiser/CUDABuffer.h
    )
    set(LIBRARIES
        ${LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${optix_LIBRARY}
    )
endif()

include_directories(${INCLUDE_DIRS})
file(GLOB PROJECT_KERNELS src/*.cl)
source_group("Kernels" FILES ${PROJECT_KERNELS})

add_executable(Fluctus ${SOURCE_FILES} ${PROJECT_KERNELS} ${PROJECT_CONFIGS})
target_link_libraries(Fluctus ${LIBRARIES})

set_target_properties(Fluctus PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib")
endif()

# Automatically copy DLLs on Windows
if (WIN32)
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(PLATFORM "Win32")
  else()
    set(PLATFORM "x64")
  endif()

  function(copydll LIBROOT LIBNAME)
    add_custom_command(TARGET Fluctus POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/ext/windows/${LIBROOT}/${PLATFORM}/lib/${LIBNAME}
        $<TARGET_FILE_DIR:Fluctus>/${LIBNAME})
  endfunction()

  copydll(DevIL DevIL.dll)
  copydll(DevIL ILU.dll)
endif()
