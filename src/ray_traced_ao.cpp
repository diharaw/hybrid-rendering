#include "ray_traced_ao.h"
#include "common_resources.h"
#include "g_buffer.h"
#include "utilities.h"
#include <profiler.h>
#include <macros.h>
#include <imgui.h>

#define NUM_THREADS_X 32
#define NUM_THREADS_Y 32

// -----------------------------------------------------------------------------------------------------------------------------------

struct AmbientOcclusionPushConstants
{
    uint32_t num_rays;
    uint32_t num_frames;
    float    ray_length;
    float    power;
    float    bias;
    uint32_t g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

RayTracedAO::RayTracedAO(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height)
{
    auto vk_backend = m_backend.lock();
    m_g_buffer_mip  = static_cast<uint32_t>(log2f(float(vk_backend->swap_chain_extents().width) / float(m_width)));

    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipeline();
}

// -----------------------------------------------------------------------------------------------------------------------------------

RayTracedAO::~RayTracedAO()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ambient Occlusion", cmd_buf);

    ray_trace(cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::gui()
{
    ImGui::PushID("RTAO");
    ImGui::Checkbox("Enabled", &m_enabled);
    ImGui::SliderInt("Num Rays", &m_num_rays, 1, 8);
    ImGui::SliderFloat("Ray Length", &m_ray_length, 1.0f, 100.0f);
    ImGui::SliderFloat("Power", &m_power, 1.0f, 5.0f);
    ImGui::InputFloat("Bias", &m_bias);
    ImGui::PopID();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::create_images()
{
    auto backend = m_backend.lock();

    m_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R8_UNORM, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_image->set_name("Ambient Occlusion Image");

    m_view = dw::vk::ImageView::create(backend, m_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_view->set_name("Ambient Occlusion Image View");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::create_descriptor_sets()
{
    auto backend = m_backend.lock();

    m_write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
    m_read_ds  = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::write_descriptor_sets()
{
    auto backend = m_backend.lock();

    // write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        image_infos.push_back(storage_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_write_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::create_pipeline()
{
    auto backend = m_backend.lock();

    dw::vk::ShaderModule::Ptr shader_module  = dw::vk::ShaderModule::create_from_file(backend, "shaders/ao_ray_trace.comp.spv");

    dw::vk::PipelineLayout::Desc pl_desc;

    pl_desc.add_descriptor_set_layout(m_common_resources->pillars_scene->descriptor_set_layout());
    pl_desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
    pl_desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
    pl_desc.add_descriptor_set_layout(m_g_buffer->ds_layout());

    pl_desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AmbientOcclusionPushConstants));

    m_pipeline_layout = dw::vk::PipelineLayout::create(backend, pl_desc);

    dw::vk::ComputePipeline::Desc desc;

    desc.set_shader_stage(shader_module, "main");
    desc.set_pipeline_layout(m_pipeline_layout);

    m_pipeline = dw::vk::ComputePipeline::create(backend, desc);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Trace", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    std::vector<VkMemoryBarrier> memory_barriers = {
        memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    pipeline_barrier(cmd_buf, memory_barriers, {}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());

    AmbientOcclusionPushConstants push_constants;

    push_constants.num_frames   = m_common_resources->num_frames;
    push_constants.num_rays     = m_num_rays;
    push_constants.ray_length   = m_ray_length;
    push_constants.power        = m_power;
    push_constants.bias         = m_bias;
    push_constants.g_buffer_mip = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene->descriptor_set()->handle(),
        m_write_ds->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_g_buffer->output_ds()->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout->handle(), 0, 4, descriptor_sets, 1, &dynamic_offset);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_width) / float(NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_height) / float(NUM_THREADS_Y))), 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::denoise(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Denoise", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::upsample(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Upsample", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::temporal_reprojection(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Temporal Reprojection", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::bilateral_blur(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Bilateral Blur", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------