cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_program(CLANG_FORMAT_EXE NAMES "clang-format" DOC "Path to clang-format executable")

add_definitions(-DDWSF_VULKAN)
add_definitions(-DDWSF_IMGUI)
add_definitions(-DDWSF_VULKAN_RAY_TRACING)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(HYBRID_RENDERING_SOURCES ${PROJECT_SOURCE_DIR}/src/main.cpp
                             ${PROJECT_SOURCE_DIR}/external/dwSampleFramework/extras/ray_traced_scene.cpp)

set(SHADER_SOURCES ${PROJECT_SOURCE_DIR}/src/shaders/g_buffer.vert
                   ${PROJECT_SOURCE_DIR}/src/shaders/g_buffer.frag
                   ${PROJECT_SOURCE_DIR}/src/shaders/copy.frag
                   ${PROJECT_SOURCE_DIR}/src/shaders/deferred.frag
                   ${PROJECT_SOURCE_DIR}/src/shaders/triangle.vert
                   ${PROJECT_SOURCE_DIR}/src/shaders/shadow.rgen
                   ${PROJECT_SOURCE_DIR}/src/shaders/shadow.rmiss
                   ${PROJECT_SOURCE_DIR}/src/shaders/shadow.rchit
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection.rgen
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection.rmiss
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection.rchit
                   ${PROJECT_SOURCE_DIR}/src/shaders/tone_map.frag
                   ${PROJECT_SOURCE_DIR}/src/shaders/svgf_reprojection.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/svgf_filter_moments.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/svgf_a_trous_filter.comp)

if(APPLE)
    add_executable(HybridRendering MACOSX_BUNDLE ${HYBRID_RENDERING_SOURCES} ${SHADER_SOURCES} ${ASSET_SOURCES})
    set(MACOSX_BUNDLE_BUNDLE_NAME "Hybrid Rendering") 
    set_source_files_properties(${SHADER_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/shader)
    set_source_files_properties(${ASSET_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
    add_executable(HybridRendering ${HYBRID_RENDERING_SOURCES} ${SHADER_SOURCES}) 
endif()

target_link_libraries(HybridRendering dwSampleFramework)

if(CLANG_FORMAT_EXE)
    add_custom_target(HybridRendering-clang-format COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${PROJECT_SOURCE_DIR}/src/main.cpp ${SHADER_SOURCES})
endif()

foreach(GLSL ${SHADER_SOURCES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${CMAKE_SOURCE_DIR}/bin/$(Configuration)/shaders/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/bin/$(Configuration)/shaders"
        COMMAND ${GLSL_VALIDATOR} --target-env vulkan1.2 -V ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(HybridRendering_Shaders DEPENDS ${SPIRV_BINARY_FILES})

add_dependencies(HybridRendering HybridRendering_Shaders)

set_property(TARGET HybridRendering PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$(Configuration)")