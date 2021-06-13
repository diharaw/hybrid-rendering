#pragma once

#include "common_resources.h"

class GBuffer;

class RayTracedAO
{
public:
    enum OutputType
    {
        OUTPUT_RAY_TRACE,
        OUTPUT_TEMPORAL_ACCUMULATION,
        OUTPUT_BILATERAL_BLUR,
        OUTPUT_DISOCCLUSION_BLUR,
        OUTPUT_UPSAMPLE
    };

    const static int         kNumOutputTypes = 5;
    const static OutputType  kOutputTypeEnums[];
    const static std::string kOutputTypeNames[];

public:
    RayTracedAO(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, RayTraceScale scale = RAY_TRACE_SCALE_HALF_RES);
    ~RayTracedAO();

    void                       render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline uint32_t      width() { return m_width; }
    inline uint32_t      height() { return m_height; }
    inline RayTraceScale scale() { return m_scale; }
    inline OutputType    current_output() { return m_current_output; }
    inline void          set_current_output(OutputType current_output) { m_current_output = current_output; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void denoise(dw::vk::CommandBuffer::Ptr cmd_buf);
    void upsample(dw::vk::CommandBuffer::Ptr cmd_buf);
    void temporal_accumulation(dw::vk::CommandBuffer::Ptr cmd_buf);
    void disocclusion_blur(dw::vk::CommandBuffer::Ptr cmd_buf);
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
        dw::vk::DescriptorSet::Ptr   bilinear_read_ds;
    };

    struct TemporalAccumulation
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

    struct DisocclusionBlur
    {
        bool                         enabled     = true;
        int32_t                      blur_radius = 2;
        int32_t                      threshold   = 15;
        dw::vk::PipelineLayout::Ptr  layout;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::Image::Ptr           image;
        dw::vk::ImageView::Ptr       image_view;
        dw::vk::DescriptorSet::Ptr   read_ds;
        dw::vk::DescriptorSet::Ptr   write_ds;
    };

    struct BilateralBlur
    {
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
    RayTraceScale                  m_scale;
    OutputType                     m_current_output = OUTPUT_UPSAMPLE;
    uint32_t                       m_width;
    uint32_t                       m_height;
    bool                           m_denoise     = true;
    bool                           m_first_frame = true;
    RayTrace                       m_ray_trace;
    TemporalAccumulation           m_temporal_accumulation;
    DisocclusionBlur               m_disocclusion_blur;
    BilateralBlur                  m_bilateral_blur;
    Upsample                       m_upsample;
};