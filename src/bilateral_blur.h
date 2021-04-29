#pragma once

#include <vk.h>
#include <glm.hpp>

class CommonResources;
class GBuffer;

class BilateralBlur
{
private:
    struct PushConstants
    {
        glm::vec4 z_buffer_params;
        float     variance_threshold;
        float     roughness_sigma_min;
        float     roughness_sigma_max;
        int32_t   radius;
        uint32_t  roughness_weight;
        uint32_t  depth_weight;
        uint32_t  normal_weight;
        uint32_t  g_buffer_mip;
    };

public:
    BilateralBlur(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height);
    ~BilateralBlur();

    void                       blur(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       prepare_first_frame(dw::vk::CommandBuffer::Ptr cmd_buf);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline int32_t blur_radius() { return m_blur_radius; }
    inline float   variance_threshold() { return m_variance_threshold; }
    inline bool    depth_weight() { return m_use_depth_weight; }
    inline bool    normal_weight() { return m_use_normal_weight; }
    inline bool    roughness_weight() { return m_use_roughness_weight; }
    inline bool    reflections_sigma_min() { return m_roughness_sigma_min; }
    inline bool    reflections_sigma_max() { return m_roughness_sigma_max; }
    inline void    set_blur_radius(int32_t n) { m_blur_radius = glm::clamp(n, 1, 7); }
    inline void    set_variance_threshold(float v) { m_variance_threshold = glm::clamp(v, 0.0f, 1.0f); }
    inline void    set_depth_weight(bool v) { m_use_depth_weight = v; }
    inline void    set_normal_weight(bool v) { m_use_normal_weight = v; }
    inline void    set_roughness_weight(bool v) { m_use_roughness_weight = v; }
    inline void    set_reflections_sigma_min(bool v) { m_roughness_sigma_min = v; }
    inline void    set_reflections_sigma_max(bool v) { m_roughness_sigma_max = v; }

private:
    std::string                    m_name;
    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    GBuffer*                       m_g_buffer;
    uint32_t                       m_input_width;
    uint32_t                       m_input_height;
    float                          m_scale                = 1.0f;
    int32_t                        m_blur_radius          = 5;
    float                          m_variance_threshold   = 0.1f;
    float                          m_roughness_sigma_min  = 0.001f;
    float                          m_roughness_sigma_max  = 0.01f;
    bool                           m_use_depth_weight     = true;
    bool                           m_use_normal_weight    = true;
    bool                           m_use_roughness_weight = true;

    // Reconstruction
    dw::vk::PipelineLayout::Ptr  m_layout;
    dw::vk::ComputePipeline::Ptr m_pipeline;
    dw::vk::Image::Ptr           m_image;
    dw::vk::ImageView::Ptr       m_image_view;
    dw::vk::DescriptorSet::Ptr   m_read_ds;
    dw::vk::DescriptorSet::Ptr   m_write_ds;
};
