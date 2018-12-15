cmake_minimum_required(VERSION 3.3)
project(Fluctus)

# OCL 1.2+ needed for clGetKernelArgInfo
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DNOMINMAX)
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

add_subdirectory(ext ext_build)

option(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
option(ASSIMP_BUILD_SAMPLES OFF)
option(ASSIMP_BUILD_TESTS OFF)
add_subdirectory(ext/assimp)

set(INCLUDE_DIRS
	ext/nanogui/include
	ext/assimp/include
	ext/glm/include
	ext/gli/include
    ${GLEW_INCLUDE_DIR}
    ${NANOGUI_EXTRA_INCS}
    ${IL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
	${Vulkan_INCLUDE_DIR}
    include
)

set(LIBRARIES
	nanogui
	assimp
    ${NANOGUI_EXTRA_LIBS}
    ${OPENGL_LIBRARIES}
    ${OpenCL_LIBRARY}
	${Vulkan_LIBRARY}
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
    src/kernelreader.cpp
    src/kernelreader.hpp
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
    src/Kernel.hpp
    src/Kernel.cpp
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
    src/utils.cpp
    src/VulkanFW/glfw/glfw.cpp
	src/VulkanFW/vks/storage.cpp
	src/VulkanFW/vks/shaders.cpp
	src/VulkanFW/vks/model.cpp
	src/VulkanFW/vks/debug.cpp
	src/VulkanFW/vks/context.cpp
    src/VulkanFW/vks/filesystem.cpp
	src/HWAccelerator.hpp
	src/HWAccelerator.cpp)

# Add configuration file if available
if (EXISTS "${CMAKE_SOURCE_DIR}/settings.json")
	file(GLOB PROJECT_CONFIGS settings.json)
endif()

# OptiX for AI denoiser, optional
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX)
if(NOT optix_LIBRARY OR NOT OptiX_INCLUDE)
	message(WARNING "OptiX not found, disabling AI denoiser.")
else()
	message(STATUS "OptiX found, building with AI denoiser.")
    add_definitions(-DWITH_OPTIX) 
	set(LIBRARIES
		${LIBRARIES}
		${optix_LIBRARY}
	)
	set(INCLUDE_DIRS
		${INCLUDE_DIRS}
		${OptiX_INCLUDE}
	)
    set(SOURCE_FILES
        ${SOURCE_FILES}
        src/OptixDenoiser.hpp
        src/OptixDenoiser.cpp
    )
endif()

include_directories(${INCLUDE_DIRS})

file(GLOB PROJECT_KERNELS src/*.cl)
source_group("Kernels" FILES ${PROJECT_KERNELS})

file(GLOB PROJECT_VK_SHADERS shaders/*.frag shaders/*.chit shaders/*.ahit)
source_group("Shaders" FILES ${PROJECT_VK_SHADERS})

add_executable(Fluctus ${SOURCE_FILES} ${PROJECT_KERNELS} ${PROJECT_VK_SHADERS} ${PROJECT_CONFIGS})
target_link_libraries(Fluctus ${LIBRARIES})

set_target_properties(Fluctus PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib")
endif()
