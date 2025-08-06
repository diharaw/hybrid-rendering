#pragma once

#include <vk.h>

struct CommonResources;

class GBuffer
{
public:
    GBuffer(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, uint32_t input_width, uint32_t input_height);
    ~GBuffer();

    void                             render(dw::vk::CommandBuffer::Ptr cmd_buf);
    dw::vk::DescriptorSetLayout::Ptr ds_layout();
    dw::vk::DescriptorSet::Ptr       output_ds();
    dw::vk::DescriptorSet::Ptr       history_ds();
    dw::vk::Image::Ptr               depth_image();
    dw::vk::ImageView::Ptr           depth_image_view();
    dw::vk::ImageView::Ptr           depth_fbo_image_view(uint32_t idx);

private:
    void create_images();
    void create_descriptor_set_layouts();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();
    void downsample_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::weak_ptr<dw::vk::Backend>   m_backend;
    CommonResources*                 m_common_resources;
    uint32_t                         m_input_width;
    uint32_t                         m_input_height;
    dw::vk::Image::Ptr               m_image_1[2]; // RGB: Albedo, A: Metallic
    dw::vk::Image::Ptr               m_image_2[2]; // RG: Normal, BA: Motion Vector
    dw::vk::Image::Ptr               m_image_3[2]; // R: Roughness, G: Curvature, B: Mesh ID, A: Linear Z
    dw::vk::Image::Ptr               m_depth[2];
    dw::vk::ImageView::Ptr           m_image_1_view[2];
    dw::vk::ImageView::Ptr           m_image_2_view[2];
    dw::vk::ImageView::Ptr           m_image_3_view[2];
    dw::vk::ImageView::Ptr           m_depth_view[2];
    dw::vk::ImageView::Ptr           m_image_1_fbo_view[2];
    dw::vk::ImageView::Ptr           m_image_2_fbo_view[2];
    dw::vk::ImageView::Ptr           m_image_3_fbo_view[2];
    dw::vk::ImageView::Ptr           m_depth_fbo_view[2];
    dw::vk::GraphicsPipeline::Ptr    m_pipeline;
    dw::vk::PipelineLayout::Ptr      m_pipeline_layout;
    dw::vk::DescriptorSetLayout::Ptr m_ds_layout;
    dw::vk::DescriptorSet::Ptr       m_ds[2];
};