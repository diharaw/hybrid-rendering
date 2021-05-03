#pragma once

#include <vk.h>
#include <glm.hpp>

class CommonResources;
class GBuffer;

class SpatialReconstruction
{
private:
    struct PushConstants
    {
        glm::vec4 z_buffer_params;
        uint32_t  num_frames;
        uint32_t  g_buffer_mip;
    };

public:
    SpatialReconstruction(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height);
    ~SpatialReconstruction();

    void                       reconstruct(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

private:
    std::string                    m_name;
    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    GBuffer*                       m_g_buffer;

    uint32_t m_input_width;
    uint32_t m_input_height;
    float    m_scale = 0.5f;

    // Reconstruction
    dw::vk::PipelineLayout::Ptr  m_layout;
    dw::vk::ComputePipeline::Ptr m_pipeline;
    dw::vk::Image::Ptr           m_image;
    dw::vk::ImageView::Ptr       m_image_view;
    dw::vk::DescriptorSet::Ptr   m_read_ds;
    dw::vk::DescriptorSet::Ptr   m_write_ds;
};
