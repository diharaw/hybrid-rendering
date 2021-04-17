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

class SVGFDenoiser;
class ReflectionDenoiser;
class DiffuseDenoiser;

struct CommonResources
{
    bool first_frame = true;
    bool ping_pong = false;
    uint32_t num_frames  = 0;
    float    near_plane  = 1.0f;
    float    far_plane   = 1000.0f;
    size_t   ubo_size    = 0;
    glm::vec4 z_buffer_params;

    // Assets.
    std::vector<dw::Mesh::Ptr> meshes;
    dw::RayTracedScene::Ptr    pillars_scene;
    dw::RayTracedScene::Ptr    sponza_scene;
    dw::RayTracedScene::Ptr    pica_pica_scene;
    dw::RayTracedScene::Ptr    current_scene;

    // Common
    dw::vk::DescriptorSet::Ptr       per_frame_ds;
    dw::vk::DescriptorSetLayout::Ptr per_frame_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr combined_sampler_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr storage_image_ds_layout;
    dw::vk::Buffer::Ptr              ubo;
    dw::vk::Image::Ptr               blue_noise_image_1;
    dw::vk::ImageView::Ptr           blue_noise_view_1;
    dw::vk::Image::Ptr               blue_noise_image_2;
    dw::vk::ImageView::Ptr           blue_noise_view_2;

    // Denoisers
    std::unique_ptr<SVGFDenoiser>       svgf_shadow_denoiser;
    std::unique_ptr<SVGFDenoiser>       svgf_gi_denoiser;
    std::unique_ptr<ReflectionDenoiser> reflection_denoiser;
    std::unique_ptr<DiffuseDenoiser>    shadow_denoiser;

    // Ray-Traced Shadows
    dw::vk::RayTracingPipeline::Ptr shadow_mask_pipeline;
    dw::vk::PipelineLayout::Ptr     shadow_mask_pipeline_layout;
    dw::vk::ShaderBindingTable::Ptr shadow_mask_sbt;
    dw::vk::Image::Ptr              visibility_image;
    dw::vk::ImageView::Ptr          visibility_view;
    dw::vk::DescriptorSet::Ptr      visibility_write_ds;
    dw::vk::DescriptorSet::Ptr      visibility_read_ds;

    // RTAO
    dw::vk::RayTracingPipeline::Ptr rtao_pipeline;
    dw::vk::PipelineLayout::Ptr     rtao_pipeline_layout;
    dw::vk::ShaderBindingTable::Ptr rtao_sbt;

    // Reflection RT pass
    dw::vk::DescriptorSet::Ptr      reflection_rt_write_ds;
    dw::vk::DescriptorSet::Ptr      reflection_rt_read_ds;
    dw::vk::RayTracingPipeline::Ptr reflection_rt_pipeline;
    dw::vk::PipelineLayout::Ptr     reflection_rt_pipeline_layout;
    dw::vk::Image::Ptr              reflection_rt_color_image;
    dw::vk::ImageView::Ptr          reflection_rt_color_view;
    dw::vk::ShaderBindingTable::Ptr reflection_rt_sbt;

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
    dw::vk::Buffer::Ptr           cube_vbo;
    dw::vk::GraphicsPipeline::Ptr skybox_pipeline;
    dw::vk::PipelineLayout::Ptr   skybox_pipeline_layout;
    dw::vk::DescriptorSet::Ptr    skybox_ds;
    dw::vk::RenderPass::Ptr       skybox_rp;
    dw::vk::Framebuffer::Ptr      skybox_fbo;

    // PBR resources
    dw::vk::DescriptorSetLayout::Ptr pbr_ds_layout;
    dw::vk::DescriptorSet::Ptr       pbr_ds;

    // Helpers
    std::unique_ptr<dw::BRDFIntegrateLUT>    brdf_preintegrate_lut;
    std::unique_ptr<dw::HosekWilkieSkyModel> hosek_wilkie_sky_model;
    std::unique_ptr<dw::CubemapSHProjection> cubemap_sh_projection;
    std::unique_ptr<dw::CubemapPrefiler>     cubemap_prefilter;
};
