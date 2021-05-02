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
    void gui();

    inline uint32_t                   width() { return m_width; }
    inline uint32_t                   height() { return m_height; }
    inline dw::vk::DescriptorSet::Ptr output_ds() { return m_read_ds; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void denoise(dw::vk::CommandBuffer::Ptr cmd_buf);
    void upsample(dw::vk::CommandBuffer::Ptr cmd_buf);
    void temporal_reprojection(dw::vk::CommandBuffer::Ptr cmd_buf);
    void bilateral_blur(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::weak_ptr<dw::vk::Backend>  m_backend;
    CommonResources*                m_common_resources;
    GBuffer*                        m_g_buffer;
    uint32_t                        m_g_buffer_mip = 0;
    uint32_t                        m_width;
    uint32_t                        m_height;
    bool                            m_enabled    = true;
    int32_t                         m_num_rays   = 2;
    float                           m_ray_length = 7.0f;
    float                           m_power      = 1.2f;
    float                           m_bias = 0.1f;
    dw::vk::ComputePipeline::Ptr m_pipeline;
    dw::vk::PipelineLayout::Ptr     m_pipeline_layout;
    dw::vk::Image::Ptr              m_image;
    dw::vk::ImageView::Ptr          m_view;
    dw::vk::DescriptorSet::Ptr      m_write_ds;
    dw::vk::DescriptorSet::Ptr      m_read_ds;
};