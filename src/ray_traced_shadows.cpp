#include "ray_traced_shadows.h"
#include "common_resources.h"
#include "g_buffer.h"
#include <profiler.h>
#include <macros.h>

struct ShadowPushConstants
{
    float    bias;
    uint32_t num_frames;
    uint32_t g_buffer_mip;
};

RayTracedShadows::RayTracedShadows(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer), m_width(width), m_height(height)
{
    auto vk_backend = m_backend.lock();
    m_g_buffer_mip  = static_cast<uint32_t>(log2f(float(vk_backend->swap_chain_extents().width) / float(m_width)));

    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipeline();
}

RayTracedShadows::~RayTracedShadows()
{

}

void RayTracedShadows::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Traced Shadows", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    // Transition ray tracing output image back to general layout

    {
        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, {}, image_barriers, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline->handle());

    ShadowPushConstants push_constants;

    push_constants.bias         = m_bias;
    push_constants.num_frames   = m_common_resources->num_frames;
    push_constants.g_buffer_mip = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene->descriptor_set()->handle(),
        m_write_ds->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_g_buffer->output_ds()->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline_layout->handle(), 0, 4, descriptor_sets, 1, &dynamic_offset);

    auto& rt_pipeline_props = backend->ray_tracing_pipeline_properties();

    VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
    VkDeviceSize group_stride = group_size;

    const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_pipeline->shader_binding_table_buffer()->device_address() + m_sbt->miss_group_offset(), group_stride, group_size * 2 };
    const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_pipeline->shader_binding_table_buffer()->device_address() + m_sbt->hit_group_offset(), group_stride, group_size * 2 };
    const VkStridedDeviceAddressRegionKHR callable_sbt = { VK_NULL_HANDLE, 0, 0 };

    vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, m_width, m_height, 1);
}

void RayTracedShadows::create_images()
{
    auto backend = m_backend.lock();

    m_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R8_UNORM, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_image->set_name("Visibility Image");
    
    m_view = dw::vk::ImageView::create(backend, m_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_view->set_name("Visibility Image View");
}

void RayTracedShadows::create_descriptor_sets()
{
    auto backend = m_backend.lock();

    m_write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
    m_read_ds  = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
}

void RayTracedShadows::write_descriptor_sets()
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

void RayTracedShadows::create_pipeline()
{
    auto backend = m_backend.lock();

    // ---------------------------------------------------------------------------
    // Create shader modules
    // ---------------------------------------------------------------------------

    dw::vk::ShaderModule::Ptr rgen  = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadow.rgen.spv");
    dw::vk::ShaderModule::Ptr rchit = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadow.rchit.spv");
    dw::vk::ShaderModule::Ptr rmiss = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadow.rmiss.spv");

    dw::vk::ShaderBindingTable::Desc sbt_desc;

    sbt_desc.add_ray_gen_group(rgen, "main");
    sbt_desc.add_hit_group(rchit, "main");
    sbt_desc.add_miss_group(rmiss, "main");

    m_sbt = dw::vk::ShaderBindingTable::create(backend, sbt_desc);

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

    pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(ShadowPushConstants));

    m_pipeline_layout = dw::vk::PipelineLayout::create(backend, pl_desc);

    desc.set_pipeline_layout(m_pipeline_layout);

    m_pipeline = dw::vk::RayTracingPipeline::create(backend, desc);
}