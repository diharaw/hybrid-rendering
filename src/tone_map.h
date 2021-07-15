#pragma once

#include <vk.h>
#include <glm.hpp>
#include <functional>

struct CommonResources;
class TemporalAA;
class DeferredShading;
class RayTracedAO;
class RayTracedShadows;
class RayTracedReflections;
class DDGI;

class ToneMap
{
public:
    ToneMap(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources);
    ~ToneMap();

    void render(dw::vk::CommandBuffer::Ptr                      cmd_buf,
                TemporalAA*                                     temporal_aa,
                DeferredShading*                                deferred_shading,
                RayTracedAO*                                    ao,
                RayTracedShadows*                               shadows,
                RayTracedReflections*                           reflections,
                DDGI*                                           ddgi,
                std::function<void(dw::vk::CommandBuffer::Ptr)> gui_callback);
    void gui();

private:
    void create_pipeline();

private:
    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    uint32_t                       m_width;
    uint32_t                       m_height;
    float                          m_exposure = 1.0f;
    dw::vk::GraphicsPipeline::Ptr  m_pipeline;
    dw::vk::PipelineLayout::Ptr    m_pipeline_layout;
};