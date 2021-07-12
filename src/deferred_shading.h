#pragma once

#include <vk.h>
#include <hosek_wilkie_sky_model.h>
#include <cubemap_sh_projection.h>
#include <cubemap_prefilter.h>
#include <mesh.h>

struct CommonResources;
class GBuffer;
class RayTracedAO;
class RayTracedShadows;
class RayTracedReflections;
class DDGI;

class DeferredShading
{
public:
    DeferredShading(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer);
    ~DeferredShading();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf,
                RayTracedAO* ao,
                RayTracedShadows* shadows,
                RayTracedReflections* reflections,
                DDGI* ddhgi);

    dw::vk::DescriptorSet::Ptr output_ds();
    dw::vk::Image::Ptr         output_image();

    inline bool use_ray_traced_ao() { return m_shading.use_ray_traced_ao; }
    inline bool use_ray_traced_shadows() { return m_shading.use_ray_traced_shadows; }
    inline bool use_ray_traced_reflections() { return m_shading.use_ray_traced_reflections; }
    inline bool use_ddgi() { return m_shading.use_ddgi; }
    inline bool visualize_probe_grid() { return m_visualize_probe_grid.enabled; }
    inline float probe_visualization_scale() { return m_visualize_probe_grid.scale; }
    inline void set_use_ray_traced_ao(bool value) { m_shading.use_ray_traced_ao = value; }
    inline void set_use_ray_traced_shadows(bool value) { m_shading.use_ray_traced_shadows = value; }
    inline void set_use_ray_traced_reflections(bool value) { m_shading.use_ray_traced_reflections = value; }
    inline void set_use_ddgi(bool value) { m_shading.use_ddgi = value; }
    inline void set_visualize_probe_grid(bool value) { m_visualize_probe_grid.enabled = value; }
    inline void set_probe_visualization_scale(float value) { m_visualize_probe_grid.scale = value; }

private:
    void load_sphere_mesh();
    void create_cube();
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_render_pass();
    void create_framebuffer();
    void create_pipeline();
    void render_shading(dw::vk::CommandBuffer::Ptr cmd_buf,
                        RayTracedAO*               ao,
                        RayTracedShadows*          shadows,
                        RayTracedReflections*      reflections,
                        DDGI*                      ddgi);
    void render_skybox(dw::vk::CommandBuffer::Ptr cmd_buf, DDGI* ddgi); 
    void render_probes(dw::vk::CommandBuffer::Ptr cmd_buf, DDGI*                      ddgi);

private:
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

    struct Shading
    {
        bool                          use_ray_traced_ao = true;
        bool                          use_ray_traced_shadows = true;
        bool                          use_ray_traced_reflections = true;
        bool                          use_ddgi = true;
        dw::vk::RenderPass::Ptr       rp;
        dw::vk::Framebuffer::Ptr      fbo;
        dw::vk::Image::Ptr            image;
        dw::vk::ImageView::Ptr        view;
        dw::vk::GraphicsPipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr   pipeline_layout;
        dw::vk::DescriptorSet::Ptr    read_ds;
    };

    struct Skybox
    {
        dw::vk::Buffer::Ptr           cube_vbo;
        dw::vk::GraphicsPipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr   pipeline_layout;
        dw::vk::RenderPass::Ptr       rp;
        dw::vk::Framebuffer::Ptr      fbo[2];
    };

    struct VisualizeProbeGrid
    {
        bool                          enabled = false;
        float                         scale   = 1.0f;
        dw::Mesh::Ptr                 sphere_mesh;
        dw::vk::GraphicsPipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr   pipeline_layout;
    };

    std::weak_ptr<dw::vk::Backend>   m_backend;
    uint32_t                         m_width;
    uint32_t                         m_height;
    CommonResources*                 m_common_resources;
    GBuffer*                         m_g_buffer;
    Shading                          m_shading;
    Skybox                           m_skybox;
    VisualizeProbeGrid m_visualize_probe_grid;
};