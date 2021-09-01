#pragma once

#include "common_resources.h"

#include <random>

class GBuffer;

class DDGI
{
public:
    DDGI(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, RayTraceScale scale = RAY_TRACE_SCALE_FULL_RES);
    ~DDGI();

    void                       render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();
    dw::vk::DescriptorSet::Ptr current_read_ds();
    uint32_t                   current_ubo_offset();

    inline uint32_t      width() { return m_width; }
    inline uint32_t      height() { return m_height; }
    inline RayTraceScale scale() { return m_scale; }
    inline glm::ivec3    probe_counts() { return m_probe_grid.probe_counts; }
    inline float         normal_bias() { return m_probe_update.normal_bias; }
    inline float         probe_distance() { return m_probe_grid.probe_distance; }
    inline float         infinite_bounce_intensity() { return m_ray_trace.infinite_bounce_intensity; }
    inline float         gi_intensity() { return m_sample_probe_grid.gi_intensity; }
    inline void          set_normal_bias(float value) { m_probe_update.normal_bias = value; }
    inline void          set_probe_distance(float value) { m_probe_grid.probe_distance = value; }
    inline void          set_infinite_bounce_intensity(float value) { m_ray_trace.infinite_bounce_intensity = value; }
    inline void          set_gi_intensity(float value) { m_sample_probe_grid.gi_intensity = value; }
    inline void          restart_accumulation() { m_first_frame = true; }

private:
    void initialize_probe_grid();
    void create_images();
    void create_buffers();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipelines();
    void recreate_probe_grid_resources();
    void update_properties_ubo();
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void probe_update(dw::vk::CommandBuffer::Ptr cmd_buf);
    void probe_update(dw::vk::CommandBuffer::Ptr cmd_buf, bool is_irradiance);
    void border_update(dw::vk::CommandBuffer::Ptr cmd_buf);
    void border_update(dw::vk::CommandBuffer::Ptr cmd_buf, bool is_irradiance);
    void sample_probe_grid(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    struct RayTrace
    {
        bool                             infinite_bounces          = true;
        float                            infinite_bounce_intensity = 1.7f;
        int32_t                          rays_per_probe            = 256;
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
        bool                             visibility_test               = true;
        float                            probe_distance                = 1.0f;
        float                            recursive_energy_preservation = 0.85f;
        uint32_t                         irradiance_oct_size           = 8;
        uint32_t                         depth_oct_size                = 16;
        glm::vec3                        grid_start_position;
        glm::ivec3                       probe_counts;
        dw::vk::DescriptorSet::Ptr       write_ds[2];
        dw::vk::DescriptorSet::Ptr       read_ds[2];
        dw::vk::DescriptorSetLayout::Ptr write_ds_layout;
        dw::vk::Image::Ptr               irradiance_image[2];
        dw::vk::Image::Ptr               depth_image[2];
        dw::vk::ImageView::Ptr           irradiance_view[2];
        dw::vk::ImageView::Ptr           depth_view[2];
        dw::vk::Buffer::Ptr              properties_ubo;
        size_t                           properties_ubo_size;
    };

    struct ProbeUpdate
    {
        float                        hysteresis      = 0.98f;
        float                        depth_sharpness = 50.0f;
        float                        max_distance    = 4.0f;
        float                        normal_bias     = 0.25f;
        dw::vk::ComputePipeline::Ptr pipeline[2];
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
    };

    struct SampleProbeGrid
    {
        float                        gi_intensity = 1.0f;
        dw::vk::Image::Ptr           image;
        dw::vk::ImageView::Ptr       image_view;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
        dw::vk::DescriptorSet::Ptr   write_ds;
        dw::vk::DescriptorSet::Ptr   read_ds;
    };

    struct BorderUpdate
    {
        dw::vk::ComputePipeline::Ptr pipeline[2];
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
    };

    uint32_t                              m_last_scene_id = UINT32_MAX;
    std::weak_ptr<dw::vk::Backend>        m_backend;
    CommonResources*                      m_common_resources;
    GBuffer*                              m_g_buffer;
    RayTraceScale                         m_scale;
    uint32_t                              m_g_buffer_mip = 0;
    uint32_t                              m_width;
    uint32_t                              m_height;
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
    SampleProbeGrid                       m_sample_probe_grid;
};