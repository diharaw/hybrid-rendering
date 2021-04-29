#pragma once

#include <vk.h>
#include <glm.hpp>

class CommonResources;
class GBuffer;

class SVGFDenoiser
{
public:
    struct ReprojectionPushConstants
    {
        float    alpha;
        float    moments_alpha;
        uint32_t g_buffer_mip;
    };

    struct FilterMomentsPushConstants
    {
        float    phi_color;
        float    phi_normal;
        uint32_t g_buffer_mip;
    };

    struct ATrousFilterPushConstants
    {
        int      radius;
        int      step_size;
        float    phi_color;
        float    phi_normal;
        uint32_t g_buffer_mip;
    };

public:
    SVGFDenoiser(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height, uint32_t filter_iterations);
    ~SVGFDenoiser();
    void                       denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    dw::vk::DescriptorSet::Ptr output_ds();

    inline uint32_t filter_iterations() { return m_a_trous_filter_iterations; }
    inline void     set_filter_iterations(uint32_t n) { m_a_trous_filter_iterations = glm::clamp(n, 1u, 5u); }

protected:
    void create_reprojection_resources();
    void create_filter_moments_resources();
    void create_a_trous_filter_resources();
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);
    void reprojection(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void filter_moments(dw::vk::CommandBuffer::Ptr cmd_buf);
    void a_trous_filter(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::string                    m_name;
    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    GBuffer*                       m_g_buffer;
    bool                           m_use_spatial_for_feedback = false;
    uint32_t                       m_input_width;
    uint32_t                       m_input_height;
    float                          m_scale                      = 1.0f;
    float                          m_alpha                      = 0.01f;
    float                          m_moments_alpha              = 0.2f;
    float                          m_phi_color                  = 10.0f;
    float                          m_phi_normal                 = 128.0f;
    int32_t                        m_a_trous_radius             = 1;
    int32_t                        m_a_trous_filter_iterations  = 4;
    int32_t                        m_a_trous_feedback_iteration = 1;
    int32_t                        m_read_idx                   = 0;

    // Reprojection
    dw::vk::ComputePipeline::Ptr     m_reprojection_pipeline;
    dw::vk::PipelineLayout::Ptr      m_reprojection_pipeline_layout;
    dw::vk::DescriptorSetLayout::Ptr m_reprojection_read_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr m_reprojection_write_ds_layout;
    dw::vk::Image::Ptr               m_reprojection_image[2];
    dw::vk::ImageView::Ptr           m_reprojection_view[2];
    dw::vk::Image::Ptr               m_moments_image[2];
    dw::vk::ImageView::Ptr           m_moments_view[2];
    dw::vk::Image::Ptr               m_history_length_image[2];
    dw::vk::ImageView::Ptr           m_history_length_view[2];
    dw::vk::DescriptorSet::Ptr       m_reprojection_write_ds[2];
    dw::vk::DescriptorSet::Ptr       m_reprojection_read_ds[2];
    dw::vk::Image::Ptr               m_prev_reprojection_image;
    dw::vk::ImageView::Ptr           m_prev_reprojection_view;
    dw::vk::DescriptorSet::Ptr       m_prev_reprojection_read_ds;

    // Filter Moments
    dw::vk::ComputePipeline::Ptr m_filter_moments_pipeline;
    dw::vk::PipelineLayout::Ptr  m_filter_moments_pipeline_layout;
    dw::vk::Image::Ptr           m_filter_moments_image;
    dw::vk::ImageView::Ptr       m_filter_moments_view;
    dw::vk::DescriptorSet::Ptr   m_filter_moments_write_ds;
    dw::vk::DescriptorSet::Ptr   m_filter_moments_read_ds;

    // A-Trous Filter
    dw::vk::ComputePipeline::Ptr m_a_trous_filter_pipeline;
    dw::vk::PipelineLayout::Ptr  m_a_trous_filter_pipeline_layout;
    dw::vk::Image::Ptr           m_a_trous_image[2];
    dw::vk::ImageView::Ptr       m_a_trous_view[2];
    dw::vk::DescriptorSet::Ptr   m_a_trous_read_ds[2];
    dw::vk::DescriptorSet::Ptr   m_a_trous_write_ds[2];
};
