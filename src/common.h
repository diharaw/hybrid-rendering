#pragma once

#include <macros.h>
#include <material.h>
#include <mesh.h>
#include <vk.h>
#include <ray_traced_scene.h>
#include <vk_mem_alloc.h>
#include <demo_player.h>
#include <brdf_preintegrate_lut.h>
#include <hosek_wilkie_sky_model.h>
#include <cubemap_sh_projection.h>
#include <cubemap_prefilter.h>
#include <stdexcept>
#include "blue_noise.h"

#define EPSILON 0.0001f
#define NUM_PILLARS 6
#define CAMERA_NEAR_PLANE 1.0f
#define CAMERA_FAR_PLANE 1000.0f
#define CAMERA_SPEED_MULTIPLIER 0.1f

class SVGFDenoiser;

namespace constants
{
extern const std::vector<std::string>            environment_map_images;
extern const std::vector<std::string>            environment_types;
extern const std::vector<std::string>            visualization_types;
extern const std::vector<std::string>            scene_types;
extern const std::vector<std::string>            ray_trace_scales;
extern const std::vector<std::string>            light_types;
extern const std::vector<std::string>            camera_types;
extern const std::vector<std::vector<glm::vec3>> fixed_camera_position_vectors;
extern const std::vector<std::vector<glm::vec3>> fixed_camera_forward_vectors;
extern const std::vector<std::vector<glm::vec3>> fixed_camera_right_vectors;
} // namespace constants

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
    SCENE_TYPE_SHADOWS_TEST,
    SCENE_TYPE_REFLECTIONS_TEST,
    SCENE_TYPE_GLOBAL_ILLUMINATION_TEST,
    SCENE_TYPE_PICA_PICA,
    SCENE_TYPE_SPONZA,
    SCENE_TYPE_COUNT
};

enum LightType
{
    LIGHT_TYPE_DIRECTIONAL,
    LIGHT_TYPE_POINT,
    LIGHT_TYPE_SPOT,
    LIGHT_TYPE_COUNT
};

enum CameraType
{
    CAMERA_TYPE_FREE,
    CAMERA_TYPE_ANIMATED,
    CAMERA_TYPE_FIXED
};

enum VisualizationType
{
    VISUALIZATION_TYPE_FINAL,
    VISUALIZATION_TYPE_SHADOWS,
    VISUALIZATION_TYPE_AMBIENT_OCCLUSION,
    VISUALIZATION_TYPE_REFLECTIONS,
    VISUALIZATION_TYPE_GLOBAL_ILLUIMINATION,
    VISUALIZATION_TYPE_GROUND_TRUTH
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

struct Light
{
    glm::vec4 data0;
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;

    inline void set_light_direction(glm::vec3 value)
    {
        data0.x = value.x;
        data0.y = value.y;
        data0.z = value.z;
    }

    inline void set_light_position(glm::vec3 value)
    {
        data1.x = value.x;
        data1.y = value.y;
        data1.z = value.z;
    }

    inline void set_light_color(glm::vec3 value)
    {
        data2.x = value.x;
        data2.y = value.y;
        data2.z = value.z;
    }

    inline void set_light_intensity(float value)
    {
        data0.w = value;
    }

    inline void set_light_radius(float value)
    {
        data1.w = value;
    }

    inline void set_light_type(LightType value)
    {
        data3.x = value;
    }

    inline void set_light_cos_theta_outer(float value)
    {
        data3.y = value;
    }

    inline void set_light_cos_theta_inner(float value)
    {
        data3.z = value;
    }
};

// Uniform buffer data structure.
struct UBO
{
    DW_ALIGNED(16)
    glm::mat4 view_inverse;
    DW_ALIGNED(16)
    glm::mat4 proj_inverse;
    DW_ALIGNED(16)
    glm::mat4 view_proj_inverse;
    DW_ALIGNED(16)
    glm::mat4 prev_view_proj;
    DW_ALIGNED(16)
    glm::mat4 view_proj;
    DW_ALIGNED(16)
    glm::vec4 cam_pos;
    DW_ALIGNED(16)
    glm::vec4 current_prev_jitter;
    DW_ALIGNED(16)
    Light light;
};

struct CommonResources
{
    SceneType                                    current_scene_type         = SCENE_TYPE_SHADOWS_TEST;
    VisualizationType                            current_visualization_type = VISUALIZATION_TYPE_FINAL;
    EnvironmentType                              current_environment_type   = ENVIRONMENT_TYPE_PROCEDURAL_SKY;
    bool                                         first_frame                = true;
    bool                                         ping_pong                  = false;
    int32_t                                      num_frames                 = 0;
    size_t                                       ubo_size                   = 0;
    glm::vec4                                    z_buffer_params;
    glm::vec3                                    camera_delta         = glm::vec3(0.0f);
    float                                        frame_time           = 0.0f;
    float                                        roughness_multiplier = 1.0f;
    glm::vec3                                    position;
    glm::vec3                                    prev_position;
    glm::mat4                                    view;
    glm::mat4                                    projection;
    glm::mat4                                    prev_view_projection;
    std::vector<std::unique_ptr<dw::DemoPlayer>> demo_players;

    // Assets.
    std::vector<dw::Mesh::Ptr>           meshes;
    std::vector<dw::RayTracedScene::Ptr> scenes;

    // Common
    dw::vk::DescriptorSet::Ptr                   per_frame_ds;
    dw::vk::DescriptorSet::Ptr                   blue_noise_ds[9];
    dw::vk::DescriptorSetLayout::Ptr             scene_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             per_frame_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             combined_sampler_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             storage_image_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr             blue_noise_ds_layout;
    dw::vk::Buffer::Ptr                          ubo;
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

    CommonResources(dw::vk::Backend::Ptr backend);
    ~CommonResources();

    void write_descriptor_sets(dw::vk::Backend::Ptr backend);

    inline dw::RayTracedScene::Ptr current_scene() { return scenes[current_scene_type]; }

private:
    void create_uniform_buffer(dw::vk::Backend::Ptr backend);
    void load_mesh(dw::vk::Backend::Ptr backend);
    void create_environment_resources(dw::vk::Backend::Ptr backend);
    void create_descriptor_set_layouts(dw::vk::Backend::Ptr backend);
    void create_descriptor_sets(dw::vk::Backend::Ptr backend);
};