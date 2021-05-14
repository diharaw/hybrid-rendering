#include "ssa_reflections.h"
#include "common_resources.h"
#include "g_buffer.h"
#include <profiler.h>
#include <macros.h>
#include <imgui.h>

#define MAX_MIP_LEVELS 1

// -----------------------------------------------------------------------------------------------------------------------------------

struct RayTracePushConstants
{
    uint32_t num_frames;
    float bias;
};

// -----------------------------------------------------------------------------------------------------------------------------------

SSaReflections::SSaReflections(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer), m_width(width), m_height(height)
{
    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipeline();
}

// -----------------------------------------------------------------------------------------------------------------------------------

SSaReflections::~SSaReflections()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("SSa Reflections", cmd_buf);

    ray_trace(cmd_buf);
    downsample(cmd_buf);
    blur(cmd_buf);
    resolve(cmd_buf);
    upsample(cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::gui()
{
    ImGui::PushID("SSaReflections");
   
    ImGui::InputFloat("Bias", &m_bias);
    
    ImGui::PopID();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Trace", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, MAX_MIP_LEVELS, 0, 1 };

    // Transition ray tracing output image back to general layout
    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_ray_trace_image->handle(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        subresource_range);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline->handle());

    RayTracePushConstants push_constants;

    push_constants.num_frames = m_common_resources->num_frames;
    push_constants.bias = m_bias;

    vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene->descriptor_set()->handle(),
        m_ray_tracing_ds->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_g_buffer->output_ds()->handle(),
        m_common_resources->pbr_ds->handle(),
        m_common_resources->skybox_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline_layout->handle(), 0, 6, descriptor_sets, 1, &dynamic_offset);

    auto& rt_pipeline_props = backend->ray_tracing_pipeline_properties();

    VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
    VkDeviceSize group_stride = group_size;

    const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_pipeline->shader_binding_table_buffer()->device_address() + m_sbt->miss_group_offset(), group_stride, group_size * 2 };
    const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_pipeline->shader_binding_table_buffer()->device_address() + m_sbt->hit_group_offset(), group_stride, group_size * 2 };
    const VkStridedDeviceAddressRegionKHR callable_sbt = { 0, 0, 0 };

    vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, m_width, m_height, 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_ray_trace_image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::downsample(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Downsample", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::blur(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Blur", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::resolve(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Resolve", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::upsample(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Upsample", cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::create_images()
{
    auto vk_backend = m_backend.lock();

    m_ray_trace_image = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, MAX_MIP_LEVELS, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_ray_trace_image->set_name("SSa Reflection RT Color Image");

    m_ray_trace_write_view = dw::vk::ImageView::create(vk_backend, m_ray_trace_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    m_ray_trace_write_view->set_name("SSa Reflection RT Color Write Image View");

    m_ray_trace_read_view = dw::vk::ImageView::create(vk_backend, m_ray_trace_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, MAX_MIP_LEVELS);
    m_ray_trace_read_view->set_name("SSa Reflection RT Color Read Image View");
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::create_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    m_ray_tracing_ds = vk_backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
    m_read_ds        = vk_backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::write_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_ray_trace_write_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        image_infos.push_back(storage_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_ray_tracing_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = vk_backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_ray_trace_read_view->handle();
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

        vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::create_pipeline()
{
    auto vk_backend = m_backend.lock();

    // ---------------------------------------------------------------------------
    // Create shader modules
    // ---------------------------------------------------------------------------

    dw::vk::ShaderModule::Ptr rgen             = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/reflection_ssa.rgen.spv");
    dw::vk::ShaderModule::Ptr rchit            = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/reflection.rchit.spv");
    dw::vk::ShaderModule::Ptr rmiss            = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/reflection.rmiss.spv");
    dw::vk::ShaderModule::Ptr rchit_visibility = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/shadow.rchit.spv");
    dw::vk::ShaderModule::Ptr rmiss_visibility = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/shadow.rmiss.spv");

    dw::vk::ShaderBindingTable::Desc sbt_desc;

    sbt_desc.add_ray_gen_group(rgen, "main");
    sbt_desc.add_hit_group(rchit, "main");
    sbt_desc.add_hit_group(rchit_visibility, "main");
    sbt_desc.add_miss_group(rmiss, "main");
    sbt_desc.add_miss_group(rmiss_visibility, "main");

    m_sbt = dw::vk::ShaderBindingTable::create(vk_backend, sbt_desc);

    dw::vk::RayTracingPipeline::Desc desc;

    desc.set_max_pipeline_ray_recursion_depth(1);
    desc.set_shader_binding_table(m_sbt);

    // ---------------------------------------------------------------------------
    // Create pipeline layout
    // ---------------------------------------------------------------------------

    dw::vk::PipelineLayout::Desc pl_desc;

    pl_desc.add_descriptor_set_layout(m_common_resources->pillars_scene->descriptor_set_layout());
    pl_desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
    pl_desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
    pl_desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
    pl_desc.add_descriptor_set_layout(m_common_resources->pbr_ds_layout);
    pl_desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
    pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(RayTracePushConstants));

    m_pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, pl_desc);

    desc.set_pipeline_layout(m_pipeline_layout);

    m_pipeline = dw::vk::RayTracingPipeline::create(vk_backend, desc);
}

// -----------------------------------------------------------------------------------------------------------------------------------