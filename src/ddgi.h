#pragma once

#include "common_resources.h"

#include <random>

class DDGI
{
public:
    DDGI(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources);
    ~DDGI();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void render_probes(dw::vk::CommandBuffer::Ptr cmd_buf);
    void gui();

    inline void  set_probe_distance(float value) { m_probe_grid.probe_distance = value; }
    inline void  set_probe_visualization_scale(float value) { m_visualize_probe_grid.scale = value; }
    inline float probe_distance() { return m_probe_grid.probe_distance; }
    inline float probe_visualization_scale() { return m_visualize_probe_grid.scale; }

private:
    void load_sphere_mesh();
    void initialize_probe_grid();
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipelines();
    void recreate_probe_grid_resources();
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void probe_update(dw::vk::CommandBuffer::Ptr cmd_buf);
    void probe_update(dw::vk::CommandBuffer::Ptr cmd_buf, bool is_irradiance);

private:
    struct RayTrace
    {
        int32_t                          rays_per_probe = 64;
        dw::vk::DescriptorSet::Ptr       write_ds;
        dw::vk::DescriptorSet::Ptr       read_ds;
        dw::vk::DescriptorSetLayout::Ptr write_ds_layout;
        dw::vk::DescriptorSetLayout::Ptr read_ds_layout;
        dw::vk::RayTracingPipeline::Ptr  pipeline;
        dw::vk::PipelineLayout::Ptr      pipeline_layout;
        dw::vk::Image::Ptr               radiance_image;
        dw::vk::Image::Ptr               direction_depth_image;
        dw::vk::ImageView::Ptr           radiance_view;
        dw::vk::ImageView::Ptr           direction_depth_view;
        dw::vk::ShaderBindingTable::Ptr  sbt;
    };

    struct ProbeGrid
    {
        float                      probe_distance      = 1.0f;
        uint32_t                   irradiance_oct_size = 8;
        uint32_t                   depth_oct_size      = 16;
        glm::vec3                  grid_start_position;
        glm::ivec3                 probe_counts;
        dw::vk::DescriptorSet::Ptr write_ds[2];
        dw::vk::DescriptorSet::Ptr read_ds[2];
        dw::vk::Image::Ptr         irradiance_image[2];
        dw::vk::Image::Ptr         depth_image[2];
        dw::vk::ImageView::Ptr     irradiance_view[2];
        dw::vk::ImageView::Ptr     depth_view[2];
    };

    struct ProbeUpdate
    {
        float                        hysteresis      = 0.98f;
        float                        depth_sharpness = 50.0f;
        float                        max_distance    = 4.0f;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
    };

    struct BorderUpdate
    {
    };

    struct VisualizeProbeGrid
    {
        bool                          enabled = false;
        float                         scale   = 1.0f;
        dw::Mesh::Ptr                 sphere_mesh;
        dw::vk::GraphicsPipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr   pipeline_layout;
    };

    uint32_t                              m_last_scene_id = UINT32_MAX;
    std::weak_ptr<dw::vk::Backend>        m_backend;
    CommonResources*                      m_common_resources;
    bool                                  m_first_frame = true;
    bool                                  m_ping_pong   = false;
    std::random_device                    m_random_device;
    std::mt19937                          m_random_generator;
    std::uniform_real_distribution<float> m_random_distribution_zo;
    std::uniform_real_distribution<float> m_random_distribution_no;
    RayTrace                              m_ray_trace;
    ProbeGrid                             m_probe_grid;
    ProbeUpdate                           m_probe_update;
    BorderUpdate                          m_border_update;
    VisualizeProbeGrid                    m_visualize_probe_grid;
};