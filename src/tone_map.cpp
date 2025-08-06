#include "tone_map.h"
#include "temporal_aa.h"
#include "deferred_shading.h"
#include "ray_traced_ao.h"
#include "ray_traced_shadows.h"
#include "ray_traced_reflections.h"
#include "ddgi.h"
#include "ground_truth_path_tracer.h"
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

void ToneMap::render(dw::vk::CommandBuffer::Ptr                      cmd_buf,
                     TemporalAA*                                     temporal_aa,
                     DeferredShading*                                deferred_shading,
                     RayTracedAO*                                    ao,
                     RayTracedShadows*                               shadows,
                     RayTracedReflections*                           reflections,
                     DDGI*                                           ddgi,
                     GroundTruthPathTracer*                          ground_truth_path_tracer,
                     std::function<void(dw::vk::CommandBuffer::Ptr)> gui_callback)
{
    DW_SCOPED_SAMPLE("Tone Map", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange input_subresource_range  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageSubresourceRange output_subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    backend->use_resource(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, backend->swapchain_image(), output_subresource_range);

    backend->flush_barriers(cmd_buf);

    VkRenderingAttachmentInfoKHR color_attachment = {};

    color_attachment.sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    color_attachment.imageView        = backend->swapchain_image_view()->handle();
    color_attachment.imageLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp           = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp          = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = { 0.0f, 0.0f, 0.0f, 0.0f };

    VkRenderingInfoKHR rendering_info {};

    rendering_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    rendering_info.renderArea           = { 0, 0, m_width, m_height };
    rendering_info.layerCount           = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments    = &color_attachment;
    rendering_info.pDepthAttachment     = nullptr;

    vkCmdBeginRenderingKHR(cmd_buf->handle(), &rendering_info);

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

    if (temporal_aa->enabled() && m_common_resources->current_visualization_type != VISUALIZATION_TYPE_GROUND_TRUTH)
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
        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_GLOBAL_ILLUIMINATION)
            read_ds = ddgi->output_ds()->handle();
        else
            read_ds = ground_truth_path_tracer->output_ds()->handle();
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

    vkCmdEndRenderingKHR(cmd_buf->handle());
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

    VkFormat format = vk_backend->swap_chain_image_format();

    m_pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
    m_pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(vk_backend, "shaders/triangle.vert.spv", "shaders/tone_map.frag.spv", m_pipeline_layout, 1, &format);
}

// -----------------------------------------------------------------------------------------------------------------------------------