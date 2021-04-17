#pragma once 

#include <vk.h>

struct CommonResources;

class GBuffer
{
public:
    GBuffer(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, uint32_t input_width, uint32_t input_height);
    ~GBuffer();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf);

    dw::vk::DescriptorSetLayout::Ptr ds_layout();
    dw::vk::DescriptorSet::Ptr       output_ds();
    dw::vk::DescriptorSet::Ptr       history_ds();
    dw::vk::ImageView::Ptr           depth_fbo_image_view();

private:    
    void create_images();
    void create_descriptor_set_layouts();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_render_pass();
    void create_framebuffer();
    void create_pipeline();
    void downsample_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf);
    void generate_mipmaps(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::Image::Ptr img, VkImageLayout src_layout, VkImageLayout dst_layout, VkFilter filter, VkImageAspectFlags aspect_flags);

private:
    std::weak_ptr<dw::vk::Backend>          m_backend;
    CommonResources*                        m_common_resources;
    uint32_t         m_input_width;
    uint32_t         m_input_height;
    dw::vk::Image::Ptr                  m_image_1; // RGB: Albedo, A: Metallic
    dw::vk::Image::Ptr                  m_image_2; // RGB: Normal, A: Roughness
    dw::vk::Image::Ptr                  m_image_3; // RGB: Position, A: -
    dw::vk::Image::Ptr                  m_image_linear_z[2];
    dw::vk::Image::Ptr                  m_depth;
    dw::vk::ImageView::Ptr              m_image_1_view;
    dw::vk::ImageView::Ptr              m_image_2_view;
    dw::vk::ImageView::Ptr              m_image_3_view;
    dw::vk::ImageView::Ptr              m_image_linear_z_view[2];
    dw::vk::ImageView::Ptr              m_depth_view;
    dw::vk::ImageView::Ptr              m_image_1_fbo_view;
    dw::vk::ImageView::Ptr              m_image_2_fbo_view;
    dw::vk::ImageView::Ptr              m_image_3_fbo_view;
    dw::vk::ImageView::Ptr              m_image_linear_z_fbo_view[2];
    dw::vk::ImageView::Ptr              m_depth_fbo_view;
    dw::vk::Framebuffer::Ptr                m_fbo[2];
    dw::vk::RenderPass::Ptr                 m_rp;
    dw::vk::GraphicsPipeline::Ptr           m_pipeline;
    dw::vk::PipelineLayout::Ptr             m_pipeline_layout;
    dw::vk::DescriptorSetLayout::Ptr        m_ds_layout;
    dw::vk::DescriptorSet::Ptr              m_ds[2];
};