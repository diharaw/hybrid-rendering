#pragma once 

#include <vk.h>

class CommonResources;
class GBuffer;

class TemporalReprojection
{
private:
    struct PushConstants
    {
        float    alpha;
        float    neighborhood_scale;
        uint32_t use_variance_clipping;
        uint32_t use_tonemap;
        uint32_t g_buffer_mip;
    };

public:
    TemporalReprojection(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height);
    ~TemporalReprojection();

    void                       reproject(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input, dw::vk::DescriptorSet::Ptr prev_input = nullptr);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline bool                             variance_clipping() { return m_use_variance_clipping; }
    inline bool                             tone_map() { return m_use_tone_map; }
    inline float                            alpha() { return m_alpha; }
    inline float                            neighborhood_scale() { return m_neighborhood_scale; }
    inline dw::vk::DescriptorSetLayout::Ptr read_ds_layout() { return m_read_ds_layout; }
    inline void                             set_variance_clipping(bool value) { m_use_tone_map = value; }
    inline void                             set_tone_map(bool value) { m_use_variance_clipping = value; }
    inline void                             set_alpha(float value) { m_alpha = value; }
    inline void                             set_neighborhood_scale(float value) { m_neighborhood_scale = value; }

private:
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::string                    m_name;
    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    GBuffer*                       m_g_buffer;
    uint32_t                       m_input_width;
    uint32_t                       m_input_height;
    float                          m_scale                 = 1.0f;
    bool                           m_use_variance_clipping = true;
    bool                           m_use_tone_map          = false;
    float                          m_neighborhood_scale    = 1.0f;
    float                          m_alpha                 = 0.01f;
    int32_t                        m_read_idx              = 0;
    // Reprojection
    dw::vk::ComputePipeline::Ptr     m_pipeline;
    dw::vk::PipelineLayout::Ptr      m_pipeline_layout;
    dw::vk::DescriptorSetLayout::Ptr m_read_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr m_write_ds_layout;
    dw::vk::Image::Ptr               m_color_image[2];
    dw::vk::ImageView::Ptr           m_color_view[2];
    dw::vk::Image::Ptr               m_history_length_image[2];
    dw::vk::ImageView::Ptr           m_history_length_view[2];
    dw::vk::DescriptorSet::Ptr       m_write_ds[2];
    dw::vk::DescriptorSet::Ptr       m_read_ds[2];
    dw::vk::DescriptorSet::Ptr       m_output_read_ds[2];
};
