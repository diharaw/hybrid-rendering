#pragma once

#include <vk.h>

struct CommonResources;
class GBuffer;

class SSaReflections
{
public:
    SSaReflections(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height);
    ~SSaReflections();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void gui();

    inline dw::vk::DescriptorSet::Ptr output_ds() { return m_read_ds; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void downsample(dw::vk::CommandBuffer::Ptr cmd_buf);
    void blur(dw::vk::CommandBuffer::Ptr cmd_buf);
    void resolve(dw::vk::CommandBuffer::Ptr cmd_buf);
    void upsample(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    GBuffer*                       m_g_buffer;
    uint32_t                       m_width;
    uint32_t                       m_height;
    float                          m_bias = 0.1f;

    dw::vk::DescriptorSet::Ptr      m_ray_tracing_ds;
    dw::vk::DescriptorSet::Ptr      m_read_ds;
    dw::vk::RayTracingPipeline::Ptr m_pipeline;
    dw::vk::PipelineLayout::Ptr     m_pipeline_layout;
    dw::vk::Image::Ptr              m_ray_trace_image;
    dw::vk::ImageView::Ptr          m_ray_trace_write_view;
    dw::vk::ImageView::Ptr          m_ray_trace_read_view;
    dw::vk::Image::Ptr              m_blurred_image;
    dw::vk::ImageView::Ptr          m_blurred_view;
    dw::vk::Image::Ptr              m_resolved_image;
    dw::vk::ImageView::Ptr          m_resolved_view;
    dw::vk::ShaderBindingTable::Ptr m_sbt;
};