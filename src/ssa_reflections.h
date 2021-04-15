#pragma once

#include <vk.h>

class HybridRendering;

class SSaReflections
{
public:
    SSaReflections(HybridRendering* sample, uint32_t width, uint32_t height);
    ~SSaReflections();

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();

private:
    HybridRendering* m_sample;
    uint32_t         m_width;
    uint32_t         m_height;

    dw::vk::DescriptorSet::Ptr      m_ray_tracing_ds;
    dw::vk::DescriptorSet::Ptr      m_read_ds;
    dw::vk::RayTracingPipeline::Ptr m_pipeline;
    dw::vk::PipelineLayout::Ptr     m_pipeline_layout;
    dw::vk::Image::Ptr              m_mirror_image;
    dw::vk::ImageView::Ptr          m_mirror_view;
    dw::vk::Image::Ptr              m_blurred_image;
    dw::vk::ImageView::Ptr          m_blurred_view;
    dw::vk::Image::Ptr              m_resolved_image;
    dw::vk::ImageView::Ptr          m_resolved_view;
    dw::vk::ShaderBindingTable::Ptr m_sbt;
};