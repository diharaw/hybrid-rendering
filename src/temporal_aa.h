#pragma once

#include <vk.h>
#include <glm.hpp>

struct CommonResources;
class GBuffer;
class DeferredShading;
class RayTracedAO;
class RayTracedShadows;
class RayTracedReflections;
class DDGI;

class TemporalAA
{
public:
    TemporalAA(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer);
    ~TemporalAA();

    void                       update();
    void                       render(dw::vk::CommandBuffer::Ptr cmd_buf,
                                      DeferredShading*           deferred_shading,
                                      RayTracedAO*               ao,
                                      RayTracedShadows*          shadows,
                                      RayTracedReflections*      reflections,
                                      DDGI*                      ddgi,
                                      float                      delta_seconds);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline bool      enabled() { return m_enabled; }
    inline glm::vec2 current_jitter() { return m_current_jitter; }
    inline glm::vec2 prev_jitter() { return m_prev_jitter; }

private:
    void create_images();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_pipeline(GBuffer* g_buffer);

private:
    std::weak_ptr<dw::vk::Backend>          m_backend;
    uint32_t                                m_width;
    uint32_t                                m_height;
    CommonResources*                        m_common_resources;
    GBuffer*                                m_g_buffer;
    std::vector<dw::vk::Image::Ptr>         m_image;
    std::vector<dw::vk::ImageView::Ptr>     m_view;
    dw::vk::ComputePipeline::Ptr            m_pipeline;
    dw::vk::PipelineLayout::Ptr             m_pipeline_layout;
    std::vector<dw::vk::DescriptorSet::Ptr> m_read_ds;
    std::vector<dw::vk::DescriptorSet::Ptr> m_write_ds;
    bool                                    m_enabled      = true;
    bool                                    m_sharpen      = true;
    bool                                    m_reset        = true;
    float                                   m_feedback_min = 0.88f;
    float                                   m_feedback_max = 0.97f;
    std::vector<glm::vec2>                  m_jitter_samples;
    glm::vec3                               m_prev_camera_pos = glm::vec3(0.0f);
    glm::vec2                               m_prev_jitter     = glm::vec2(0.0f);
    glm::vec2                               m_current_jitter  = glm::vec2(0.0f);
};