cmake_minimum_required(VERSION 3.3)
project(Fluctus)

add_subdirectory(ext ext_build)

include_directories(
    ext/nanogui/include
    ${GLEW_INCLUDE_DIR}
    ${NANOGUI_EXTRA_INCS}
    ${IL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    include
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
    src/envmap.hpp
    src/envmap.cpp
    src/rgbe/rgbe.hpp
    src/rgbe/rgbe.cpp
    src/settings.cpp
    src/settings.hpp
    src/texture.cpp
    src/texture.hpp
    src/GLProgram.cpp
    src/GLProgram.hpp
    src/utils.h
    src/bvh.cl
    src/env_map.cl
    src/random.cl
    src/utils.cl
    src/intersect.cl
	src/fresnel.cl
	src/ideal_reflection.cl
	src/ideal_dielectric.cl
	src/glossy.cl
	src/ggx.cl
	src/diffuse.cl
	src/bxdf.cl
    src/kernel_monolith.cl
    src/mk_next_vertex.cl
    src/mk_raygen.cl
    src/mk_reset.cl
    src/mk_sample_bsdf.cl
    src/mk_splat.cl
    src/mk_splat_preview.cl)

add_executable(Fluctus ${SOURCE_FILES})

target_link_libraries(Fluctus
    nanogui
    ${NANOGUI_EXTRA_LIBS}
    ${OPENGL_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${IL_LIBRARIES}
    ${ILU_LIBRARIES}
)

set_target_properties(Fluctus PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib")
endif()
