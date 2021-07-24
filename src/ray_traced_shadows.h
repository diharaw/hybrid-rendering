#pragma once

#include "common_resources.h"

class GBuffer;

class RayTracedShadows
{
public:
    enum OutputType
    {
        OUTPUT_RAY_TRACE,
        OUTPUT_TEMPORAL_ACCUMULATION,
        OUTPUT_ATROUS,
        OUTPUT_UPSAMPLE
    };

    const static int         kNumOutputTypes = 4;
    const static OutputType  kOutputTypeEnums[];
    const static std::string kOutputTypeNames[];

public:
    RayTracedShadows(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, RayTraceScale scale = RAY_TRACE_SCALE_FULL_RES);
    ~RayTracedShadows();

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
    void create_buffers();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipelines();
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void reset_args(dw::vk::CommandBuffer::Ptr cmd_buf);
    void temporal_accumulation(dw::vk::CommandBuffer::Ptr cmd_buf);
    void a_trous_filter(dw::vk::CommandBuffer::Ptr cmd_buf);
    void upsample(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    struct RayTrace
    {
        float                        bias = 0.5f;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
        dw::vk::Image::Ptr           image;
        dw::vk::ImageView::Ptr       view;
        dw::vk::DescriptorSet::Ptr   write_ds;
        dw::vk::DescriptorSet::Ptr   read_ds;
    };

    struct ResetArgs
    {
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
        dw::vk::ComputePipeline::Ptr pipeline;
    };

    struct TemporalAccumulation
    {
        float                            alpha         = 0.01f;
        float                            moments_alpha = 0.2f;
        dw::vk::Buffer::Ptr              denoise_tile_coords_buffer;
        dw::vk::Buffer::Ptr              denoise_dispatch_args_buffer;
        dw::vk::Buffer::Ptr              shadow_tile_coords_buffer;
        dw::vk::Buffer::Ptr              shadow_dispatch_args_buffer;
        dw::vk::ComputePipeline::Ptr     pipeline;
        dw::vk::PipelineLayout::Ptr      pipeline_layout;
        dw::vk::DescriptorSetLayout::Ptr write_ds_layout;
        dw::vk::DescriptorSetLayout::Ptr read_ds_layout;
        dw::vk::DescriptorSetLayout::Ptr indirect_buffer_ds_layout;
        dw::vk::Image::Ptr               current_output_image;
        dw::vk::Image::Ptr               current_moments_image[2];
        dw::vk::Image::Ptr               prev_image;
        dw::vk::ImageView::Ptr           current_output_view;
        dw::vk::ImageView::Ptr           current_moments_view[2];
        dw::vk::ImageView::Ptr           prev_view;
        dw::vk::DescriptorSet::Ptr       current_write_ds[2];
        dw::vk::DescriptorSet::Ptr       current_read_ds[2];
        dw::vk::DescriptorSet::Ptr       output_only_read_ds;
        dw::vk::DescriptorSet::Ptr       prev_read_ds[2];
        dw::vk::DescriptorSet::Ptr       indirect_buffer_ds;
    };

    struct CopyShadowTiles
    {
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
        dw::vk::ComputePipeline::Ptr pipeline;
    };

    struct ATrous
    {
        float                        phi_visibility     = 10.0f;
        float                        phi_normal         = 32.0f;
        float                        sigma_depth        = 1.0f;
        float                        power              = 1.2f;
        int32_t                      radius             = 1;
        int32_t                      filter_iterations  = 3;
        int32_t                      feedback_iteration = 1;
        int32_t                      read_idx           = 0;
        dw::vk::ComputePipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
        dw::vk::Image::Ptr           image[2];
        dw::vk::ImageView::Ptr       view[2];
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
    uint32_t                       m_g_buffer_mip   = 0;
    uint32_t                       m_width;
    uint32_t                       m_height;
    bool                           m_denoise     = true;
    bool                           m_first_frame = true;
    RayTrace                       m_ray_trace;
    ResetArgs                      m_reset_args;
    TemporalAccumulation           m_temporal_accumulation;
    CopyShadowTiles                m_copy_shadow_tiles;
    ATrous                         m_a_trous;
    Upsample                       m_upsample;
};