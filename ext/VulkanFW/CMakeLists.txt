set(TARGET_NAME base)

set(SHADER_DIR "${PROJECT_SOURCE_DIR}/data/shaders")
file(GLOB_RECURSE SHADERS
    ${SHADER_DIR}/*.vert
    ${SHADER_DIR}/*.tesc
    ${SHADER_DIR}/*.tese
    ${SHADER_DIR}/*.geom
    ${SHADER_DIR}/*.frag
    ${SHADER_DIR}/*.comp
    ${SHADER_DIR}/*.mesh
    ${SHADER_DIR}/*.task
    ${SHADER_DIR}/*.rgen
    ${SHADER_DIR}/*.rint
    ${SHADER_DIR}/*.rahit
    ${SHADER_DIR}/*.rchit
    ${SHADER_DIR}/*.rmiss
    ${SHADER_DIR}/*.rcall
    ${SHADER_DIR}/*.hlsl
)
GroupSources("data/shaders")


GroupSources("base")
foreach(SHADER ${SHADERS})
    compile_spirv_shader(${SHADER})
    list(APPEND COMPILED_SHADERS ${COMPILE_SPIRV_SHADER_RETURN})
    source_group("compiled" FILES ${COMPILE_SPIRV_SHADER_RETURN})
endforeach()
add_custom_target(shaders SOURCES ${SHADERS} ${COMPILED_SHADERS})
set_target_properties(shaders PROPERTIES FOLDER "common")

file(GLOB_RECURSE COMMON_SOURCE *.c *.cpp *.h *.hpp)
add_library(${TARGET_NAME} STATIC ${COMMON_SOURCE})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "common")
add_dependencies(${TARGET_NAME} shaders)
target_link_libraries(${TARGET_NAME} imgui)

target_glfw3()
target_glm()
target_gli()
target_vulkan()
target_assimp()
target_imgui()
target_glad()

if (ANDROID)
    add_dependencies(${TARGET_NAME} app-glue)
    target_link_libraries(${TARGET_NAME} app-glue android log m)
endif()