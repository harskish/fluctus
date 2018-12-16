cmake_minimum_required(VERSION 3.6)
project(GLAD C)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
    # Ensure we generate position independent code                                                                                                                                                                                            
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    endif()
endif()

add_library(glad STATIC src/glad.c)
set_target_properties(glad PROPERTIES DEBUG_POSTFIX _d)

file(GLOB GLAD_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/glad/*.h)

install(TARGETS glad DESTINATION lib)
install(FILES include/glad/glad.h DESTINATION include/glad)
install(FILES include/KHR/khrplatform.h DESTINATION include/KHR)
