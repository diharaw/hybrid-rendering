#pragma once

#include <vk.h>

struct CommonResources;
class GBuffer;

class RayTracedAO
{
public:
    RayTracedAO(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer);
    ~RayTracedAO();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void gui();

    inline uint32_t                   width() { return m_width; }
    inline uint32_t                   height() { return m_height; }
    inline dw::vk::DescriptorSet::Ptr output_ds() { return m_upsample.read_ds; }
    inline bool                       enabled() { return m_enabled; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void denoise(dw::vk::CommandBuffer::Ptr cmd_buf);
    void upsample(dw::vk::CommandBuffer::Ptr cmd_buf);
    void temporal_reprojection(dw::vk::CommandBuffer::Ptr cmd_buf);
    void bilateral_blur(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    struct RayTrace
    {
        int32_t                      num_rays   = 2;
        float                        ray_length = 7.0f;
        float                        power      = 1.2f;
        float                        bias       = 0.3f;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
        dw::vk::Image::Ptr           image;
        dw::vk::ImageView::Ptr       view;
        dw::vk::DescriptorSet::Ptr   write_ds;
        dw::vk::DescriptorSet::Ptr   read_ds;
    };

    struct TemporalReprojection
    {
        float                            alpha = 0.01f;
        dw::vk::ComputePipeline::Ptr     pipeline;
        dw::vk::PipelineLayout::Ptr      pipeline_layout;
        dw::vk::DescriptorSetLayout::Ptr read_ds_layout;
        dw::vk::DescriptorSetLayout::Ptr write_ds_layout;
        dw::vk::Image::Ptr               color_image[2];
        dw::vk::ImageView::Ptr           color_view[2];
        dw::vk::Image::Ptr               history_length_image[2];
        dw::vk::ImageView::Ptr           history_length_view[2];
        dw::vk::DescriptorSet::Ptr       write_ds[2];
        dw::vk::DescriptorSet::Ptr       read_ds[2];
        dw::vk::DescriptorSet::Ptr       output_read_ds[2];
    };

    struct BilateralBlur
    {
        bool                         recurrent   = false;
        int32_t                      blur_radius = 5;
        dw::vk::PipelineLayout::Ptr  layout;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::Image::Ptr           image[2];
        dw::vk::ImageView::Ptr       image_view[2];
        dw::vk::DescriptorSet::Ptr   read_ds[2];
        dw::vk::DescriptorSet::Ptr   write_ds[2];
    };

    struct Upsample
    {
        dw::vk::PipelineLayout::Ptr  layout;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::Image::Ptr           image;
        dw::vk::ImageView::Ptr       image_view;
        dw::vk::DescriptorSet::Ptr   read_ds;
        dw::vk::DescriptorSet::Ptr   write_ds;
    };

    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    GBuffer*                       m_g_buffer;
    uint32_t                       m_width;
    uint32_t                       m_height;
    bool                           m_enabled = true;
    RayTrace                       m_ray_trace;
    TemporalReprojection           m_temporal_reprojection;
    BilateralBlur                  m_bilateral_blur;
    Upsample                       m_upsample;
};