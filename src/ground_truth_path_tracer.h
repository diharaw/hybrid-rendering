#pragma once

#include "common.h"

class GBuffer;

class GroundTruthPathTracer
{
public:
    GroundTruthPathTracer(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources);
    ~GroundTruthPathTracer();

    void                       render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline void restart_accumulation() { m_frame_idx = 0; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipelines();

private:
    struct PathTrace
    {
        int32_t                         max_ray_bounces = 2;
        dw::vk::DescriptorSet::Ptr      write_ds[2];
        dw::vk::DescriptorSet::Ptr      read_ds[2];
        dw::vk::RayTracingPipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr     pipeline_layout;
        dw::vk::Image::Ptr              images[2];
        dw::vk::ImageView::Ptr          image_views[2];
        dw::vk::ShaderBindingTable::Ptr sbt;
    };

    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    uint32_t                       m_width;
    uint32_t                       m_height;
    uint32_t                       m_frame_idx = 0;
    bool                           m_ping_pong = false;
    PathTrace                      m_path_trace;
};