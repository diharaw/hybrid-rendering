#pragma once

#include <material.h>
#include <mesh.h>
#include <vk.h>
#include <ray_traced_scene.h>
#include <vk_mem_alloc.h>
#include <brdf_preintegrate_lut.h>
#include <hosek_wilkie_sky_model.h>
#include <cubemap_sh_projection.h>
#include <cubemap_prefilter.h>
#include <stdexcept>
#include "blue_noise.h"

class SVGFDenoiser;

enum RayTraceScale
{
    RAY_TRACE_SCALE_FULL_RES,
    RAY_TRACE_SCALE_HALF_RES,
    RAY_TRACE_SCALE_QUARTER_RES
};

enum EnvironmentType
{
    ENVIRONMENT_TYPE_NONE,
    ENVIRONMENT_TYPE_PROCEDURAL_SKY,
    ENVIRONMENT_TYPE_ARCHES_PINE_TREE,
    ENVIRONMENT_TYPE_BASKETBALL_COURT,
    ENVIRONMENT_TYPE_ETNIES_PART_CENTRAL,
    ENVIRONMENT_TYPE_LA_DOWNTOWN_HELIPAD
};

enum SceneType
{
    SCENE_TYPE_PILLARS,
    SCENE_TYPE_SPONZA,
    SCENE_TYPE_PICA_PICA,
    SCENE_TYPE_COUNT
};

enum LightType
{
    LIGHT_TYPE_DIRECTIONAL,
    LIGHT_TYPE_POINT
};

enum VisualizationType
{
    VISUALIZATION_TYPE_FINAL,
    VISUALIZATION_TYPE_SHADOWS,
    VISUALIZATION_TYPE_AMBIENT_OCCLUSION,
    VISUALIZATION_TYPE_REFLECTIONS,
    VISUALIZATION_TYPE_GLOBAL_ILLUIMINATION
};

struct SkyEnvironment
{
    std::unique_ptr<dw::CubemapSHProjection> cubemap_sh_projection;
    std::unique_ptr<dw::CubemapPrefiler>     cubemap_prefilter;
    std::unique_ptr<dw::HosekWilkieSkyModel> hosek_wilkie_sky_model;
};

struct HDREnvironment
{
    dw::vk::Image::Ptr                       image;
    dw::vk::ImageView::Ptr                   image_view;
    std::unique_ptr<dw::CubemapSHProjection> cubemap_sh_projection;
    std::unique_ptr<dw::CubemapPrefiler>     cubemap_prefilter;
};

struct CommonResources
{
    SceneType         current_scene_type         = SCENE_TYPE_PILLARS;
    VisualizationType current_visualization_type = VISUALIZATION_TYPE_FINAL;
    EnvironmentType   current_environment_type   = ENVIRONMENT_TYPE_PROCEDURAL_SKY;
    bool              first_frame                = true;
    bool              ping_pong                  = false;
    int32_t           num_frames                 = 0;
    size_t            ubo_size                   = 0;
    glm::vec4         z_buffer_params;
    glm::vec3         camera_delta = glm::vec3(0.0f);
    float             frame_time   = 0.0f;
    glm::vec3         position;
    glm::vec3         prev_position;
    glm::mat4         view;
    glm::mat4         projection;
    glm::mat4         prev_view_projection;

    // Assets.
    std::vector<dw::Mesh::Ptr>           meshes;
    std::vector<dw::RayTracedScene::Ptr> scenes;

    // Common
    dw::vk::DescriptorSet::Ptr                   per_frame_ds;
    dw::vk::DescriptorSet::Ptr                   blue_noise_ds[9];
    dw::vk::DescriptorSetLayout::Ptr             per_frame_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             combined_sampler_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             storage_image_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             blue_noise_ds_layout;
    dw::vk::Buffer::Ptr                          ubo;
    dw::vk::Image::Ptr                           blue_noise_image_1;
    dw::vk::ImageView::Ptr                       blue_noise_view_1;
    dw::vk::Image::Ptr                           blue_noise_image_2;
    dw::vk::ImageView::Ptr                       blue_noise_view_2;
    dw::vk::Buffer::Ptr                          bnd_sobol_buffer;
    dw::vk::Buffer::Ptr                          bnd_scrambling_tile_buffer;
    dw::vk::Buffer::Ptr                          bnd_ranking_tile_buffer;
    std::unique_ptr<BlueNoise>                   blue_noise;
    dw::vk::DescriptorSetLayout::Ptr             ddgi_read_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             skybox_ds_layout;
    std::vector<dw::vk::DescriptorSet::Ptr>      skybox_ds;
    dw::vk::DescriptorSet::Ptr                   current_skybox_ds;
    dw::vk::Image::Ptr                           blank_sh_image;
    dw::vk::ImageView::Ptr                       blank_sh_image_view;
    dw::vk::Image::Ptr                           blank_cubemap_image;
    dw::vk::ImageView::Ptr                       blank_cubemap_image_view;
    std::unique_ptr<SkyEnvironment>              sky_environment;
    std::vector<std::shared_ptr<HDREnvironment>> hdr_environments;
    std::unique_ptr<dw::BRDFIntegrateLUT>        brdf_preintegrate_lut;

    inline dw::RayTracedScene::Ptr current_scene() { return scenes[current_scene_type]; }
};
