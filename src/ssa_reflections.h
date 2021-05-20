#pragma once

#include <vk.h>

struct CommonResources;
class GBuffer;

class SSaReflections
{
public:
    SSaReflections(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height);
    ~SSaReflections();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void gui();

    inline dw::vk::DescriptorSet::Ptr output_ds() { return m_blur.read_ds; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline();
    void ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf);
    void image_pyramid(dw::vk::CommandBuffer::Ptr cmd_buf);
    void blur(dw::vk::CommandBuffer::Ptr cmd_buf);
    void resolve(dw::vk::CommandBuffer::Ptr cmd_buf);
    void upsample(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    struct RayTrace
    {
        dw::vk::DescriptorSetLayout::Ptr              write_ds_layout;
        dw::vk::DescriptorSet::Ptr write_ds;
        dw::vk::DescriptorSet::Ptr      read_ds;
        dw::vk::RayTracingPipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr     pipeline_layout;
        dw::vk::Image::Ptr              color_image;
        dw::vk::Image::Ptr                      hit_distance_image;
        std::vector<dw::vk::ImageView::Ptr>           single_color_image_views;
        dw::vk::ImageView::Ptr          all_color_image_view;
        dw::vk::ImageView::Ptr                  hit_distance_image_view;
        dw::vk::ShaderBindingTable::Ptr sbt;
    };

    struct ImagePyramid
    {
        std::vector<dw::vk::DescriptorSet::Ptr> write_ds;
        std::vector<dw::vk::DescriptorSet::Ptr> read_ds;
        dw::vk::PipelineLayout::Ptr  pipeline_layout;
        dw::vk::ComputePipeline::Ptr pipeline;
    };

    struct Blur
    {
        std::vector<dw::vk::DescriptorSet::Ptr> write_ds;
        dw::vk::DescriptorSet::Ptr              read_ds;
        dw::vk::PipelineLayout::Ptr             pipeline_layout;
        dw::vk::ComputePipeline::Ptr            pipeline;
        dw::vk::Image::Ptr                      image;
        std::vector<dw::vk::ImageView::Ptr>     single_image_views;
        dw::vk::ImageView::Ptr                  all_image_view;
    };

    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    GBuffer*                       m_g_buffer;
    uint32_t                       m_width;
    uint32_t                       m_height;
    float                          m_bias = 0.5f;
    RayTrace m_ray_trace;
    ImagePyramid                    m_image_pyramid;
    Blur                            m_blur;
};