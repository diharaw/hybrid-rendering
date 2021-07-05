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
    bool      first_frame = true;
    bool      ping_pong   = false;
    int32_t   num_frames  = 0;
    size_t    ubo_size    = 0;
    glm::vec4 z_buffer_params;
    glm::vec3 camera_delta = glm::vec3(0.0f);
    float     frame_time   = 0.0f;

    // Assets.
    std::vector<dw::Mesh::Ptr> meshes;
    dw::RayTracedScene::Ptr    pillars_scene;
    dw::RayTracedScene::Ptr    sponza_scene;
    dw::RayTracedScene::Ptr    pica_pica_scene;
    dw::RayTracedScene::Ptr    current_scene;

    // Common
    dw::vk::DescriptorSet::Ptr       per_frame_ds;
    dw::vk::DescriptorSet::Ptr       blue_noise_ds[9];
    dw::vk::DescriptorSetLayout::Ptr per_frame_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr combined_sampler_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr storage_image_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr blue_noise_ds_layout;
    dw::vk::Buffer::Ptr              ubo;
    dw::vk::Image::Ptr               blue_noise_image_1;
    dw::vk::ImageView::Ptr           blue_noise_view_1;
    dw::vk::Image::Ptr               blue_noise_image_2;
    dw::vk::ImageView::Ptr           blue_noise_view_2;
    dw::vk::Buffer::Ptr              bnd_sobol_buffer;
    dw::vk::Buffer::Ptr              bnd_scrambling_tile_buffer;
    dw::vk::Buffer::Ptr              bnd_ranking_tile_buffer;
    std::unique_ptr<BlueNoise>       blue_noise;

    // Denoisers
    std::unique_ptr<SVGFDenoiser> svgf_gi_denoiser;

    // Global Illumination Ray Tracing pass
    dw::vk::DescriptorSet::Ptr      rtgi_write_ds;
    dw::vk::DescriptorSet::Ptr      rtgi_read_ds;
    dw::vk::RayTracingPipeline::Ptr rtgi_pipeline;
    dw::vk::PipelineLayout::Ptr     rtgi_pipeline_layout;
    dw::vk::Image::Ptr              rtgi_image;
    dw::vk::ImageView::Ptr          rtgi_view;
    dw::vk::ShaderBindingTable::Ptr rtgi_sbt;

    // Deferred pass
    dw::vk::RenderPass::Ptr       deferred_rp;
    dw::vk::Framebuffer::Ptr      deferred_fbo;
    dw::vk::Image::Ptr            deferred_image;
    dw::vk::ImageView::Ptr        deferred_view;
    dw::vk::GraphicsPipeline::Ptr deferred_pipeline;
    dw::vk::PipelineLayout::Ptr   deferred_pipeline_layout;
    dw::vk::DescriptorSet::Ptr    deferred_read_ds;

    // TAA pass
    std::vector<dw::vk::Image::Ptr>         taa_image;
    std::vector<dw::vk::ImageView::Ptr>     taa_view;
    dw::vk::ComputePipeline::Ptr            taa_pipeline;
    dw::vk::PipelineLayout::Ptr             taa_pipeline_layout;
    std::vector<dw::vk::DescriptorSet::Ptr> taa_read_ds;
    std::vector<dw::vk::DescriptorSet::Ptr> taa_write_ds;

    // Copy pass
    dw::vk::GraphicsPipeline::Ptr copy_pipeline;
    dw::vk::PipelineLayout::Ptr   copy_pipeline_layout;

    // Skybox
    dw::vk::Buffer::Ptr                          cube_vbo;
    dw::vk::GraphicsPipeline::Ptr                skybox_pipeline;
    dw::vk::PipelineLayout::Ptr                  skybox_pipeline_layout;
    dw::vk::DescriptorSetLayout::Ptr             skybox_ds_layout;
    std::vector<dw::vk::DescriptorSet::Ptr>      skybox_ds;
    dw::vk::DescriptorSet::Ptr                   current_skybox_ds;
    dw::vk::RenderPass::Ptr                      skybox_rp;
    dw::vk::Framebuffer::Ptr                     skybox_fbo[2];
    dw::vk::Image::Ptr                           blank_sh_image;
    dw::vk::ImageView::Ptr                       blank_sh_image_view;
    dw::vk::Image::Ptr                           blank_cubemap_image;
    dw::vk::ImageView::Ptr                       blank_cubemap_image_view;
    std::unique_ptr<SkyEnvironment>              sky_environment;
    std::vector<std::shared_ptr<HDREnvironment>> hdr_environments;

    // Helpers
    std::unique_ptr<dw::BRDFIntegrateLUT> brdf_preintegrate_lut;
};
