#pragma once

#include <vk.h>

struct CommonResources;
class GBuffer;

class RayTracedShadows
{
public:
    RayTracedShadows(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height);
    ~RayTracedShadows();

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();

private:
    std::weak_ptr<dw::vk::Backend>  m_backend;
    CommonResources*                m_common_resources;
    GBuffer*                        m_g_buffer;
    uint32_t                        m_width;
    uint32_t                        m_height;
    float   m_bias              = 0.1f;
    dw::vk::RayTracingPipeline::Ptr m_pipeline;
    dw::vk::PipelineLayout::Ptr     m_pipeline_layout;
    dw::vk::ShaderBindingTable::Ptr m_sbt;
    dw::vk::Image::Ptr              m_image;
    dw::vk::ImageView::Ptr          m_view;
    dw::vk::DescriptorSet::Ptr      m_write_ds;
    dw::vk::DescriptorSet::Ptr      m_read_ds;
};