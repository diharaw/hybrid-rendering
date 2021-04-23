#pragma once

#include <vk.h>

struct CommonResources;
class GBuffer;

class RayTracedAO
{
public:
    RayTracedAO(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height);
    ~RayTracedAO();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf);

    inline uint32_t                   width() { return m_width; }
    inline uint32_t                   height() { return m_height; }
    inline dw::vk::DescriptorSet::Ptr output_ds() { return m_read_ds; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();

private:
    std::weak_ptr<dw::vk::Backend>  m_backend;
    CommonResources*                m_common_resources;
    GBuffer*                        m_g_buffer;
    uint32_t                        m_g_buffer_mip = 0;
    uint32_t                        m_width;
    uint32_t                        m_height;
    int32_t                         m_num_rays   = 2;
    float                           m_ray_length = 30.0f;
    float                           m_power      = 5.0f;
    float                           m_bias = 0.1f;
    dw::vk::RayTracingPipeline::Ptr m_pipeline;
    dw::vk::PipelineLayout::Ptr     m_pipeline_layout;
    dw::vk::ShaderBindingTable::Ptr m_sbt;
    dw::vk::Image::Ptr              m_image;
    dw::vk::ImageView::Ptr          m_view;
    dw::vk::DescriptorSet::Ptr      m_write_ds;
    dw::vk::DescriptorSet::Ptr      m_read_ds;
};