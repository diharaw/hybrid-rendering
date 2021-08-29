#include "ground_truth_path_tracer.h"
#include "utilities.h"
#include <profiler.h>
#include <macros.h>
#include <imgui.h>

// -----------------------------------------------------------------------------------------------------------------------------------

struct PathTracePushConstants
{
    uint32_t num_frames;
    uint32_t max_ray_bounces;
};

// -----------------------------------------------------------------------------------------------------------------------------------

GroundTruthPathTracer::GroundTruthPathTracer(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources) :
    m_backend(backend), m_common_resources(common_resources)
{
    auto vk_backend = m_backend.lock();

    m_width  = vk_backend->swap_chain_extents().width;
    m_height = vk_backend->swap_chain_extents().height;

    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipelines();
}

// -----------------------------------------------------------------------------------------------------------------------------------

GroundTruthPathTracer::~GroundTruthPathTracer()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GroundTruthPathTracer::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_GROUND_TRUTH)
    {
        DW_SCOPED_SAMPLE("Ground Truth Path Trace", cmd_buf);

        if (m_frame_idx == 0)
            m_ping_pong = false;

        auto backend = m_backend.lock();

        const uint32_t read_idx  = static_cast<uint32_t>(m_ping_pong);
        const uint32_t write_idx = static_cast<uint32_t>(!m_ping_pong);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        {
            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_path_trace.images[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
                image_memory_barrier(m_path_trace.images[read_idx], m_frame_idx == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
        }

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_path_trace.pipeline->handle());

        PathTracePushConstants push_constants;

        push_constants.num_frames      = m_frame_idx++;
        push_constants.max_ray_bounces = m_path_trace.max_ray_bounces;

        vkCmdPushConstants(cmd_buf->handle(), m_path_trace.pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offsets[] = {
            m_common_resources->ubo_size * backend->current_frame_idx()
        };

        VkDescriptorSet descriptor_sets[] = {
            m_common_resources->current_scene()->descriptor_set()->handle(),
            m_path_trace.write_ds[write_idx]->handle(),
            m_path_trace.write_ds[read_idx]->handle(),
            m_common_resources->per_frame_ds->handle(),
            m_common_resources->current_skybox_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_path_trace.pipeline_layout->handle(), 0, 5, descriptor_sets, 1, dynamic_offsets);

        auto& rt_pipeline_props = backend->ray_tracing_pipeline_properties();

        VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
        VkDeviceSize group_stride = group_size;

        const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_path_trace.pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
        const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_path_trace.pipeline->shader_binding_table_buffer()->device_address() + m_path_trace.sbt->miss_group_offset(), group_stride, group_size };
        const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_path_trace.pipeline->shader_binding_table_buffer()->device_address() + m_path_trace.sbt->hit_group_offset(), group_stride, group_size };
        const VkStridedDeviceAddressRegionKHR callable_sbt = { 0, 0, 0 };

        uint32_t rt_image_width  = m_width;
        uint32_t rt_image_height = m_height;

        vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, rt_image_width, rt_image_height, 1);

        {
            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_path_trace.images[write_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }

        m_ping_pong = !m_ping_pong;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GroundTruthPathTracer::gui()
{
    ImGui::SliderInt("Path Trace Bounces", &m_path_trace.max_ray_bounces, 1, 5);
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::DescriptorSet::Ptr GroundTruthPathTracer::output_ds()
{
    return m_path_trace.read_ds[static_cast<uint32_t>(m_ping_pong)];
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GroundTruthPathTracer::create_images()
{
    auto backend = m_backend.lock();

    for (int i = 0; i < 2; i++)
    {
        m_path_trace.images[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_path_trace.images[i]->set_name("Ground Truth Path Trace");

        m_path_trace.image_views[i] = dw::vk::ImageView::create(backend, m_path_trace.images[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_path_trace.image_views[i]->set_name("Ground Truth Path Trace");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GroundTruthPathTracer::create_descriptor_sets()
{
    auto backend = m_backend.lock();

    for (int i = 0; i < 2; i++)
    {
        m_path_trace.write_ds[i] = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_path_trace.read_ds[i]  = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GroundTruthPathTracer::write_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Path Trace Write
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
            storage_image_info.imageView   = m_path_trace.image_views[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_path_trace.write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Path Trace Read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        for (int i = 0; i < 2; i++)
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_path_trace.image_views[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_path_trace.read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GroundTruthPathTracer::create_pipelines()
{
    auto backend = m_backend.lock();

    // ---------------------------------------------------------------------------
    // Create shader modules
    // ---------------------------------------------------------------------------

    dw::vk::ShaderModule::Ptr rgen  = dw::vk::ShaderModule::create_from_file(backend, "shaders/ground_truth_path_trace.rgen.spv");
    dw::vk::ShaderModule::Ptr rchit = dw::vk::ShaderModule::create_from_file(backend, "shaders/ground_truth_path_trace.rchit.spv");
    dw::vk::ShaderModule::Ptr rmiss = dw::vk::ShaderModule::create_from_file(backend, "shaders/ground_truth_path_trace.rmiss.spv");

    dw::vk::ShaderBindingTable::Desc sbt_desc;

    sbt_desc.add_ray_gen_group(rgen, "main");
    sbt_desc.add_hit_group(rchit, "main");
    sbt_desc.add_miss_group(rmiss, "main");

    m_path_trace.sbt = dw::vk::ShaderBindingTable::create(backend, sbt_desc);

    dw::vk::RayTracingPipeline::Desc desc;

    desc.set_max_pipeline_ray_recursion_depth(8);
    desc.set_shader_binding_table(m_path_trace.sbt);

    // ---------------------------------------------------------------------------
    // Create pipeline layout
    // ---------------------------------------------------------------------------

    dw::vk::PipelineLayout::Desc pl_desc;

    pl_desc.add_descriptor_set_layout(m_common_resources->current_scene()->descriptor_set_layout());
    pl_desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
    pl_desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
    pl_desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
    pl_desc.add_descriptor_set_layout(m_common_resources->skybox_ds_layout);
    pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(PathTracePushConstants));

    m_path_trace.pipeline_layout = dw::vk::PipelineLayout::create(backend, pl_desc);

    desc.set_pipeline_layout(m_path_trace.pipeline_layout);

    m_path_trace.pipeline = dw::vk::RayTracingPipeline::create(backend, desc);
}

// -----------------------------------------------------------------------------------------------------------------------------------