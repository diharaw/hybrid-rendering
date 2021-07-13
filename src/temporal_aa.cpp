#include "temporal_aa.h"
#include "g_buffer.h"
#include "deferred_shading.h"
#include "ray_traced_ao.h"
#include "ray_traced_shadows.h"
#include "ray_traced_reflections.h"
#include "ddgi.h"
#include "utilities.h"
#include <imgui.h>
#include <profiler.h>
#include <macros.h>
#include <GLFW/glfw3.h>

#define HALTON_SAMPLES 16

// -----------------------------------------------------------------------------------------------------------------------------------

struct TAAPushConstants
{
    glm::vec4 texel_size;
    glm::vec4 current_prev_jitter;
    glm::vec4 time_params;
    float     feedback_min;
    float     feedback_max;
    int       sharpen;
};

// -----------------------------------------------------------------------------------------------------------------------------------

float halton_sequence(int base, int index)
{
    float result = 0;
    float f      = 1;
    while (index > 0)
    {
        f /= base;
        result += f * (index % base);
        index = floor(index / base);
    }

    return result;
}

// -----------------------------------------------------------------------------------------------------------------------------------

TemporalAA::TemporalAA(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer)
{
    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipeline(g_buffer);

    for (int i = 1; i <= HALTON_SAMPLES; i++)
        m_jitter_samples.push_back(glm::vec2((2.0f * halton_sequence(2, i) - 1.0f), (2.0f * halton_sequence(3, i) - 1.0f)));
}

// -----------------------------------------------------------------------------------------------------------------------------------

TemporalAA::~TemporalAA()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TemporalAA::update()
{
    if (m_enabled)
    {
        m_prev_jitter        = m_current_jitter;
        uint32_t  sample_idx = m_common_resources->num_frames % (m_jitter_samples.size());
        glm::vec2 halton     = m_jitter_samples[sample_idx];

        m_current_jitter = glm::vec2(halton.x / float(m_width), halton.y / float(m_height));
    }
    else
    {
        m_prev_jitter    = glm::vec2(0.0f);
        m_current_jitter = glm::vec2(0.0f);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TemporalAA::render(dw::vk::CommandBuffer::Ptr cmd_buf,
                        DeferredShading*           deferred_shading,
                        RayTracedAO*               ao,
                        RayTracedShadows*          shadows,
                        RayTracedReflections*      reflections,
                        DDGI*                      ddgi,
                        float                      delta_seconds)
{
    if (m_enabled)
    {
        DW_SCOPED_SAMPLE("TAA", cmd_buf);

        const uint32_t NUM_THREADS = 32;
        const uint32_t write_idx   = (uint32_t)m_common_resources->ping_pong;
        const uint32_t read_idx    = (uint32_t)!m_common_resources->ping_pong;

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                            m_image[write_idx]->handle(),
                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_GENERAL,
                                            subresource_range);

        if (m_reset)
        {
            dw::vk::utilities::blitt_image(cmd_buf,
                                           deferred_shading->output_image(),
                                           m_image[read_idx],
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           VK_IMAGE_LAYOUT_UNDEFINED,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           VK_IMAGE_ASPECT_COLOR_BIT,
                                           VK_FILTER_NEAREST);
        }

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());

        TAAPushConstants push_constants;

        push_constants.texel_size          = glm::vec4(1.0f / float(m_width), 1.0f / float(m_height), float(m_width), float(m_height));
        push_constants.current_prev_jitter = glm::vec4(m_current_jitter, m_prev_jitter);
        push_constants.time_params         = glm::vec4(static_cast<float>(glfwGetTime()), sinf(static_cast<float>(glfwGetTime())), cosf(static_cast<float>(glfwGetTime())), delta_seconds);
        push_constants.feedback_min        = m_feedback_min;
        push_constants.feedback_max        = m_feedback_max;
        push_constants.sharpen             = static_cast<int>(m_sharpen);

        vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet read_ds;

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

        VkDescriptorSet descriptor_sets[] = {
            m_write_ds[write_idx]->handle(),
            read_ds,
            m_read_ds[read_idx]->handle(),
            m_g_buffer->output_ds()->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(),
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipeline_layout->handle(),
                                0,
                                4,
                                descriptor_sets,
                                0,
                                nullptr);

        vkCmdDispatch(cmd_buf->handle(),
                      static_cast<uint32_t>(ceil(float(m_width) / float(NUM_THREADS))),
                      static_cast<uint32_t>(ceil(float(m_height) / float(NUM_THREADS))),
                      1);

        // Prepare ray tracing output image as transfer source
        dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                            m_image[write_idx]->handle(),
                                            VK_IMAGE_LAYOUT_GENERAL,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                            subresource_range);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TemporalAA::gui()
{
    ImGui::PushID("GUI_TAA");
    if (ImGui::Checkbox("Enabled", &m_enabled))
    {
        if (m_enabled)
            m_reset = true;
    }
    ImGui::Checkbox("Sharpen", &m_sharpen);
    ImGui::SliderFloat("Feedback Min", &m_feedback_min, 0.0f, 1.0f);
    ImGui::SliderFloat("Feedback Max", &m_feedback_max, 0.0f, 1.0f);
    ImGui::PopID();
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::DescriptorSet::Ptr TemporalAA::output_ds()
{
    return m_read_ds[m_common_resources->ping_pong];
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TemporalAA::create_images()
{
    auto vk_backend = m_backend.lock();

    m_width  = vk_backend->swap_chain_extents().width;
    m_height = vk_backend->swap_chain_extents().height;

    // TAA
    for (int i = 0; i < 2; i++)
    {
        auto image = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
        image->set_name("TAA Image " + std::to_string(i));

        m_image.push_back(image);

        auto image_view = dw::vk::ImageView::create(vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        image_view->set_name("TAA Image View " + std::to_string(i));

        m_view.push_back(image_view);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TemporalAA::create_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    for (int i = 0; i < 2; i++)
    {
        m_read_ds.push_back(vk_backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout));
        m_write_ds.push_back(vk_backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout));
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TemporalAA::write_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    // TAA read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        for (int i = 0; i < 2; i++)
        {
            VkDescriptorImageInfo combined_sampler_image_info;

            combined_sampler_image_info.sampler     = vk_backend->bilinear_sampler()->handle();
            combined_sampler_image_info.imageView   = m_view[i]->handle();
            combined_sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(combined_sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // TAA write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        for (int i = 0; i < 2; i++)
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TemporalAA::create_pipeline(GBuffer* g_buffer)
{
    auto vk_backend = m_backend.lock();

    dw::vk::PipelineLayout::Desc desc;

    desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
    desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
    desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
    desc.add_descriptor_set_layout(g_buffer->ds_layout());
    desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TAAPushConstants));

    m_pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);

    dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/taa.comp.spv");

    dw::vk::ComputePipeline::Desc comp_desc;

    comp_desc.set_pipeline_layout(m_pipeline_layout);
    comp_desc.set_shader_stage(module, "main");

    m_pipeline = dw::vk::ComputePipeline::create(vk_backend, comp_desc);
}

// -----------------------------------------------------------------------------------------------------------------------------------