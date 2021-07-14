#include "tone_map.h"
#include "temporal_aa.h"
#include "deferred_shading.h"
#include "ray_traced_ao.h"
#include "ray_traced_shadows.h"
#include "ray_traced_reflections.h"
#include "ddgi.h"
#include "utilities.h"
#include <imgui.h>
#include <profiler.h>
#include <macros.h>

// -----------------------------------------------------------------------------------------------------------------------------------

struct ToneMapPushConstants
{
    int   single_channel;
    float exposure;
};

// -----------------------------------------------------------------------------------------------------------------------------------

ToneMap::ToneMap(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources) :
    m_backend(backend), m_common_resources(common_resources)
{
    auto vk_backend = backend.lock();

    m_width  = vk_backend->swap_chain_extents().width;
    m_height = vk_backend->swap_chain_extents().height;

    create_pipeline();
}

// -----------------------------------------------------------------------------------------------------------------------------------

ToneMap::~ToneMap()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMap::render(dw::vk::CommandBuffer::Ptr cmd_buf,
            TemporalAA*                temporal_aa,
            DeferredShading*           deferred_shading,
            RayTracedAO*               ao,
            RayTracedShadows*          shadows,
            RayTracedReflections*      reflections,
                     DDGI*                      ddgi,
                     std::function<void(dw::vk::CommandBuffer::Ptr)> gui_callback)
{
    DW_SCOPED_SAMPLE("Tone Map", cmd_buf);

    auto vk_backend = m_backend.lock();

    VkClearValue clear_values[2];

    clear_values[0].color.float32[0] = 0.0f;
    clear_values[0].color.float32[1] = 0.0f;
    clear_values[0].color.float32[2] = 0.0f;
    clear_values[0].color.float32[3] = 1.0f;

    clear_values[1].color.float32[0] = 1.0f;
    clear_values[1].color.float32[1] = 1.0f;
    clear_values[1].color.float32[2] = 1.0f;
    clear_values[1].color.float32[3] = 1.0f;

    VkRenderPassBeginInfo info    = {};
    info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass               = vk_backend->swapchain_render_pass()->handle();
    info.framebuffer              = vk_backend->swapchain_framebuffer()->handle();
    info.renderArea.extent.width  = m_width;
    info.renderArea.extent.height = m_height;
    info.clearValueCount          = 2;
    info.pClearValues             = &clear_values[0];

    vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp;

    vp.x        = 0.0f;
    vp.y        = (float)m_height;
    vp.width    = (float)m_width;
    vp.height   = -(float)m_height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

    VkRect2D scissor_rect;

    scissor_rect.extent.width  = m_width;
    scissor_rect.extent.height = m_height;
    scissor_rect.offset.x      = 0;
    scissor_rect.offset.y      = 0;

    vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());

    VkDescriptorSet read_ds;

    if (temporal_aa->enabled())
        read_ds = temporal_aa->output_ds()->handle();
    else
    {
        if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_FINAL)
            read_ds = deferred_shading->output_ds()->handle();
        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_SHADOWS)
            read_ds = shadows->output_ds()->handle();
        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_AMBIENT_OCCLUSION)
            read_ds = ao->output_ds()->handle();
        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_REFLECTIONS)
            read_ds = reflections->output_ds()->handle();
        else
            read_ds = ddgi->output_ds()->handle();
    }

    VkDescriptorSet descriptor_sets[] = {
        read_ds
    };

    ToneMapPushConstants push_constants;

    push_constants.single_channel = (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_SHADOWS || m_common_resources->current_visualization_type == VISUALIZATION_TYPE_AMBIENT_OCCLUSION) ? 1 : 0;
    push_constants.exposure       = m_exposure;

    vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ToneMapPushConstants), &push_constants);

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout->handle(), 0, 1, descriptor_sets, 0, nullptr);

    vkCmdDraw(cmd_buf->handle(), 3, 1, 0, 0);

    if (gui_callback)
        gui_callback(cmd_buf);

    vkCmdEndRenderPass(cmd_buf->handle());
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMap::gui()
{
    ImGui::InputFloat("Exposure", &m_exposure);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMap::create_pipeline()
{
    auto vk_backend = m_backend.lock();

    dw::vk::PipelineLayout::Desc desc;

    desc.add_push_constant_range(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ToneMapPushConstants));
    desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);

    m_pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
    m_pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(vk_backend, "shaders/triangle.vert.spv", "shaders/tone_map.frag.spv", m_pipeline_layout, vk_backend->swapchain_render_pass());
}

// -----------------------------------------------------------------------------------------------------------------------------------