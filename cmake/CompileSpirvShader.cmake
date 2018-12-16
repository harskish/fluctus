# 
#  CompileSpirvShader.cmake
# 
#  Created by Bradley Austin Davis on 2016/06/23
#  Modified by Erik Härkönen on 2018/12/17
#
#  Distributed under the Apache License, Version 2.0.
#  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
# 

# Always forces rebuild
add_custom_command(
    OUTPUT always_rebuild
    COMMAND cmake -E echo
)

function(COMPILE_SPIRV_SHADER SHADER_FILE)
    # Define the final name of the generated shader file
    find_program(GLSLANG_EXECUTABLE glslangValidator
        HINTS "$ENV{VULKAN_SDK}/bin")
    get_filename_component(SHADER_TARGET ${SHADER_FILE} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)
    set(COMPILE_OUTPUT "${SHADER_FILE}.spv")
    add_custom_command(
        OUTPUT ${COMPILE_OUTPUT}
        COMMAND ${GLSLANG_EXECUTABLE} -V ${SHADER_FILE} -o ${COMPILE_OUTPUT}
        DEPENDS always_rebuild)
    set(COMPILE_SPIRV_SHADER_RETURN ${COMPILE_OUTPUT} PARENT_SCOPE)
endfunction()

