cmake_minimum_required(VERSION 3.3)
project(BMFR)

# OCL 1.2+ needed for clGetKernelArgInfo
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DNOMINMAX)

find_package(OpenCL 1.2 REQUIRED)
find_package(OpenGL REQUIRED)

set(LIBRARIES
    ${OPENGL_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${CMAKE_SOURCE_DIR}\\..\\..\\ext\\windows\\oiio\\lib\\x64\\OpenImageIOD.lib
    
    # C:\\Users\\Erik\\code\\fluctus\\ext\\windows\\oiio\\lib\\x64\\OpenImageIOD.lib
)

set(INCLUDE_DIRS
	${OpenCL_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
	../../ext/windows/oiio/include
    CLUtils
)

set(SOURCE_FILES
    bmfr.cpp
    CLUtils/CLUtils.cpp
)

include_directories(${INCLUDE_DIRS})
file(GLOB PROJECT_KERNELS *.cl)
source_group("Kernels" FILES ${PROJECT_KERNELS})

add_executable(BMFR ${SOURCE_FILES} ${PROJECT_KERNELS})
target_link_libraries(BMFR ${LIBRARIES})

set_target_properties(BMFR PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_custom_command(TARGET BMFR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../../ext/windows/oiio/bin/x64/OpenImageIOD.dll $<TARGET_FILE_DIR:BMFR>/OpenImageIOD.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../../ext/windows/msvcp110d.dll $<TARGET_FILE_DIR:BMFR>/msvcp110d.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../../ext/windows/msvcr110d.dll $<TARGET_FILE_DIR:BMFR>/msvcr110d.dll
    
)
