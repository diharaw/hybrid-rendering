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
                             ${PROJECT_SOURCE_DIR}/src/ddgi.cpp
                             ${PROJECT_SOURCE_DIR}/src/ground_truth_path_tracer.cpp
                             ${PROJECT_SOURCE_DIR}/src/path_traced_gi.cpp
                             ${PROJECT_SOURCE_DIR}/src/ray_traced_ao.cpp
                             ${PROJECT_SOURCE_DIR}/src/ray_traced_shadows.cpp
                             ${PROJECT_SOURCE_DIR}/src/ssa_reflections.cpp
                             ${PROJECT_SOURCE_DIR}/src/stochastic_reflections.cpp
                             ${PROJECT_SOURCE_DIR}/src/g_buffer.cpp
                             ${PROJECT_SOURCE_DIR}/src/deferred_shading.cpp
                             ${PROJECT_SOURCE_DIR}/src/temporal_aa.cpp
                             ${PROJECT_SOURCE_DIR}/src/tonemap.cpp
                             ${PROJECT_SOURCE_DIR}/src/utilities.cpp
                             ${PROJECT_SOURCE_DIR}/src/temporal_reprojection.cpp
                             ${PROJECT_SOURCE_DIR}/src/spatial_reconstruction.cpp
                             ${PROJECT_SOURCE_DIR}/src/bilateral_blur.cpp
                             ${PROJECT_SOURCE_DIR}/src/svgf_denoiser.cpp
                             ${PROJECT_SOURCE_DIR}/src/diffuse_denoiser.cpp
                             ${PROJECT_SOURCE_DIR}/src/reflection_denoiser.cpp
                             ${PROJECT_SOURCE_DIR}/src/blue_noise.cpp
                             ${PROJECT_SOURCE_DIR}/src/common_resources.h
                             ${PROJECT_SOURCE_DIR}/src/ddgi.h
                             ${PROJECT_SOURCE_DIR}/src/ground_truth_path_tracer.h
                             ${PROJECT_SOURCE_DIR}/src/path_traced_gi.h
                             ${PROJECT_SOURCE_DIR}/src/ray_traced_ao.h
                             ${PROJECT_SOURCE_DIR}/src/ray_traced_shadows.h
                             ${PROJECT_SOURCE_DIR}/src/ssa_reflections.h
                             ${PROJECT_SOURCE_DIR}/src/stochastic_reflections.h
                             ${PROJECT_SOURCE_DIR}/src/g_buffer.h
                             ${PROJECT_SOURCE_DIR}/src/deferred_shading.h
                             ${PROJECT_SOURCE_DIR}/src/temporal_aa.h
                             ${PROJECT_SOURCE_DIR}/src/tonemap.h
                             ${PROJECT_SOURCE_DIR}/src/utilities.h
                             ${PROJECT_SOURCE_DIR}/src/temporal_reprojection.h
                             ${PROJECT_SOURCE_DIR}/src/spatial_reconstruction.h
                             ${PROJECT_SOURCE_DIR}/src/bilateral_blur.h
                             ${PROJECT_SOURCE_DIR}/src/svgf_denoiser.h
                             ${PROJECT_SOURCE_DIR}/src/diffuse_denoiser.h
                             ${PROJECT_SOURCE_DIR}/src/reflection_denoiser.h
                             ${PROJECT_SOURCE_DIR}/src/blue_noise.h
                             ${PROJECT_SOURCE_DIR}/external/dwSampleFramework/extras/brdf_preintegrate_lut.cpp
                             ${PROJECT_SOURCE_DIR}/external/dwSampleFramework/extras/cubemap_prefilter.cpp
                             ${PROJECT_SOURCE_DIR}/external/dwSampleFramework/extras/cubemap_sh_projection.cpp
                             ${PROJECT_SOURCE_DIR}/external/dwSampleFramework/extras/hosek_wilkie_sky_model.cpp
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
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection_ssa.rgen
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection.rmiss
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection.rchit
                   ${PROJECT_SOURCE_DIR}/src/shaders/ambient_occlusion.rgen
                   ${PROJECT_SOURCE_DIR}/src/shaders/indirect_diffuse.rgen
                   ${PROJECT_SOURCE_DIR}/src/shaders/indirect_diffuse.rmiss
                   ${PROJECT_SOURCE_DIR}/src/shaders/indirect_diffuse.rchit
                   ${PROJECT_SOURCE_DIR}/src/shaders/tone_map.frag
                   ${PROJECT_SOURCE_DIR}/src/shaders/svgf_reprojection.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/svgf_filter_moments.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/svgf_a_trous_filter.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/skybox.vert
                   ${PROJECT_SOURCE_DIR}/src/shaders/skybox.frag
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection_spatial_resolve.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection_reprojection.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/reflection_blur.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/taa.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/gi_reprojection.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/bilateral_blur.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/temporal_reprojection.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/spatial_reconstruction.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/ssa_image_pyramid.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/ao_ray_trace.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/ao_denoise_reprojection.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/ao_denoise_downsample.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/ao_denoise_gaussian_blur.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/ao_denoise_recurrent_blur.comp
                   ${PROJECT_SOURCE_DIR}/src/shaders/ao_upsample.comp)

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
    add_custom_target(HybridRendering-clang-format COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${HYBRID_RENDERING_SOURCES} ${SHADER_SOURCES})
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