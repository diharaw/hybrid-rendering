#include "ray_traced_reflections.h"
#include "g_buffer.h"
#include "utilities.h"
#include <profiler.h>
#include <macros.h>
#include <imgui.h>

struct RayTracePushConstants
{
    float    bias;
    float    trim;
    uint32_t num_frames;
    int32_t  g_buffer_mip;
};

struct TemporalAccumulationPushConstants
{
    glm::vec3 camera_delta;
    float     frame_time;
    float     alpha;
    float     moments_alpha;
    int32_t   g_buffer_mip;
};

struct ATrousFilterPushConstants
{
    int     radius;
    int     step_size;
    float   phi_color;
    float   phi_normal;
    int32_t g_buffer_mip;
};

const RayTracedReflections::Output RayTracedReflections::kOutputTypeEnums[] = {
    RayTracedReflections::OUTPUT_RAY_TRACE,
    RayTracedReflections::OUTPUT_TEMPORAL_ACCUMULATION,
    RayTracedReflections::OUTPUT_ATROUS
};

const std::string RayTracedReflections::kOutputTypeNames[] = {
    "Ray Trace",
    "Temporal Accumulation",
    "A-Trous"
};

RayTracedReflections::RayTracedReflections(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, RayTraceScale scale) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer), m_scale(scale)
{
    auto vk_backend = m_backend.lock();

    float scale_divisor = powf(2.0f, float(scale));

    m_width  = vk_backend->swap_chain_extents().width / scale_divisor;
    m_height = vk_backend->swap_chain_extents().height / scale_divisor;

    m_g_buffer_mip = static_cast<uint32_t>(scale);

    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipelines();
}

RayTracedReflections::~RayTracedReflections()
{
}

void RayTracedReflections::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Traced Reflections", cmd_buf);

    clear_images(cmd_buf);
    ray_trace(cmd_buf);

    if (m_denoise)
    {
        temporal_accumulation(cmd_buf);
        a_trous_filter(cmd_buf);
    }
}

void RayTracedReflections::gui()
{
    ImGui::Checkbox("Denoise", &m_denoise);
    ImGui::InputFloat("Bias", &m_ray_trace.bias);
    ImGui::SliderFloat("Lobe Trim", &m_ray_trace.trim, 0.0f, 1.0f);
    ImGui::InputFloat("Alpha", &m_temporal_accumulation.alpha);
    ImGui::InputFloat("Alpha Moments", &m_temporal_accumulation.moments_alpha);
    ImGui::InputFloat("Phi Color", &m_a_trous.phi_color);
    ImGui::InputFloat("Phi Normal", &m_a_trous.phi_normal);
}

dw::vk::DescriptorSet::Ptr RayTracedReflections::output_ds(Output output_type)
{
    if (m_denoise)
    {
        if (output_type == OUTPUT_RAY_TRACE)
            return m_ray_trace.read_ds;
        else if (output_type == OUTPUT_TEMPORAL_ACCUMULATION)
            return m_temporal_accumulation.output_only_read_ds;
        else
            return m_a_trous.read_ds[m_a_trous.read_idx];
    }
    else
        return m_ray_trace.read_ds;
}

void RayTracedReflections::create_images()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        m_ray_trace.image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_ray_trace.image->set_name("Reflections Ray Trace");

        m_ray_trace.view = dw::vk::ImageView::create(backend, m_ray_trace.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_ray_trace.view->set_name("Reflections Ray Trace");
    }

    // Reprojection
    {
        m_temporal_accumulation.current_output_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_temporal_accumulation.current_output_image->set_name("Reflections Reprojection Output");

        m_temporal_accumulation.current_output_view = dw::vk::ImageView::create(backend, m_temporal_accumulation.current_output_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_temporal_accumulation.current_output_view->set_name("Reflections Reprojection Output");

        for (int i = 0; i < 2; i++)
        {
            m_temporal_accumulation.current_moments_image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
            m_temporal_accumulation.current_moments_image[i]->set_name("Reflections Reprojection Moments " + std::to_string(i));

            m_temporal_accumulation.current_moments_view[i] = dw::vk::ImageView::create(backend, m_temporal_accumulation.current_moments_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            m_temporal_accumulation.current_moments_view[i]->set_name("Reflections Reprojection Moments " + std::to_string(i));
        }

        m_temporal_accumulation.prev_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_temporal_accumulation.prev_image->set_name("Reflections Previous Reprojection");

        m_temporal_accumulation.prev_view = dw::vk::ImageView::create(backend, m_temporal_accumulation.prev_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_temporal_accumulation.prev_view->set_name("Reflections Previous Reprojection");
    }

    // A-Trous Filter
    for (int i = 0; i < 2; i++)
    {
        m_a_trous.image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_a_trous.image[i]->set_name("A-Trous Filter " + std::to_string(i));

        m_a_trous.view[i] = dw::vk::ImageView::create(backend, m_a_trous.image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_a_trous.view[i]->set_name("A-Trous Filter View " + std::to_string(i));
    }
}

void RayTracedReflections::create_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        m_ray_trace.write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_ray_trace.read_ds  = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
    }

    // Reprojection
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_temporal_accumulation.write_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_temporal_accumulation.read_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_temporal_accumulation.current_write_ds[i] = backend->allocate_descriptor_set(m_temporal_accumulation.write_ds_layout);
        m_temporal_accumulation.current_read_ds[i]  = backend->allocate_descriptor_set(m_temporal_accumulation.read_ds_layout);
        m_temporal_accumulation.prev_read_ds[i]     = backend->allocate_descriptor_set(m_temporal_accumulation.read_ds_layout);
    }

    m_temporal_accumulation.output_only_read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);

    // A-Trous
    for (int i = 0; i < 2; i++)
    {
        m_a_trous.read_ds[i]  = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_a_trous.write_ds[i] = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
    }
}

void RayTracedReflections::write_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Ray Trace Write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_ray_trace.view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        image_infos.push_back(storage_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_ray_trace.write_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Ray Trace Read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_ray_trace.view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_ray_trace.read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection Output Only Read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_temporal_accumulation.current_output_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_temporal_accumulation.output_only_read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection Current Write
    for (int i = 0; i < 2; i++)
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_temporal_accumulation.current_output_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_temporal_accumulation.current_write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_temporal_accumulation.current_moments_view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_temporal_accumulation.current_write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection Current Read
    for (int i = 0; i < 2; i++)
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_temporal_accumulation.current_output_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_temporal_accumulation.current_read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_temporal_accumulation.current_moments_view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_temporal_accumulation.current_read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection Prev Read
    for (int i = 0; i < 2; i++)
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_temporal_accumulation.prev_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_temporal_accumulation.prev_read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_temporal_accumulation.current_moments_view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_temporal_accumulation.prev_read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // A-Trous write
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
            storage_image_info.imageView   = m_a_trous.view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_a_trous.write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
    // A-Trous read
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
            sampler_image_info.imageView   = m_a_trous.view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_a_trous.read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void RayTracedReflections::create_pipelines()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr rgen  = dw::vk::ShaderModule::create_from_file(backend, "shaders/reflections_ray_trace.rgen.spv");
        dw::vk::ShaderModule::Ptr rchit = dw::vk::ShaderModule::create_from_file(backend, "shaders/reflections_ray_trace.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss = dw::vk::ShaderModule::create_from_file(backend, "shaders/reflections_ray_trace.rmiss.spv");

        dw::vk::ShaderBindingTable::Desc sbt_desc;

        sbt_desc.add_ray_gen_group(rgen, "main");
        sbt_desc.add_hit_group(rchit, "main");
        sbt_desc.add_miss_group(rmiss, "main");

        m_ray_trace.sbt = dw::vk::ShaderBindingTable::create(backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_max_pipeline_ray_recursion_depth(1);
        desc.set_shader_binding_table(m_ray_trace.sbt);

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
        pl_desc.add_descriptor_set_layout(m_common_resources->blue_noise_ds_layout);
        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(RayTracePushConstants));

        m_ray_trace.pipeline_layout = dw::vk::PipelineLayout::create(backend, pl_desc);

        desc.set_pipeline_layout(m_ray_trace.pipeline_layout);

        m_ray_trace.pipeline = dw::vk::RayTracingPipeline::create(backend, desc);
    }

    // Reprojection
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_temporal_accumulation.write_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_temporal_accumulation.read_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TemporalAccumulationPushConstants));

        m_temporal_accumulation.pipeline_layout = dw::vk::PipelineLayout::create(backend, desc);
        m_temporal_accumulation.pipeline_layout->set_name("Reprojection Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/reflections_denoise_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_temporal_accumulation.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_temporal_accumulation.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // A-Trous Filter
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ATrousFilterPushConstants));

        m_a_trous.pipeline_layout = dw::vk::PipelineLayout::create(backend, desc);
        m_a_trous.pipeline_layout->set_name("A-Trous Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/reflections_denoise_atrous.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_a_trous.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_a_trous.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }
}

void RayTracedReflections::clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    if (m_first_frame)
    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkClearColorValue color;

        color.float32[0] = 0.0f;
        color.float32[1] = 0.0f;
        color.float32[2] = 0.0f;
        color.float32[3] = 0.0f;

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temporal_accumulation.prev_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temporal_accumulation.current_moments_image[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdClearColorImage(cmd_buf->handle(), m_temporal_accumulation.prev_image->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_temporal_accumulation.current_moments_image[!m_common_resources->ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temporal_accumulation.prev_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temporal_accumulation.current_moments_image[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        m_first_frame = false;
    }
}

void RayTracedReflections::ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Trace", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    std::vector<VkMemoryBarrier> memory_barriers = {
        memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    std::vector<VkImageMemoryBarrier> image_barriers = {
        image_memory_barrier(m_ray_trace.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
    };

    pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_ray_trace.pipeline->handle());

    RayTracePushConstants push_constants;

    push_constants.bias         = m_ray_trace.bias;
    push_constants.trim         = m_ray_trace.trim;
    push_constants.num_frames   = m_common_resources->num_frames;
    push_constants.g_buffer_mip = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_ray_trace.pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene->descriptor_set()->handle(),
        m_ray_trace.write_ds->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_g_buffer->output_ds()->handle(),
        m_common_resources->pbr_ds->handle(),
        m_common_resources->skybox_ds->handle(),
        m_common_resources->blue_noise_ds[BLUE_NOISE_1SPP]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_ray_trace.pipeline_layout->handle(), 0, 7, descriptor_sets, 1, &dynamic_offset);

    auto& rt_pipeline_props = backend->ray_tracing_pipeline_properties();

    VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
    VkDeviceSize group_stride = group_size;

    const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address() + m_ray_trace.sbt->miss_group_offset(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address() + m_ray_trace.sbt->hit_group_offset(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR callable_sbt = { 0, 0, 0 };

    uint32_t rt_image_width  = m_width;
    uint32_t rt_image_height = m_height;

    vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, rt_image_width, rt_image_height, 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_ray_trace.image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

void RayTracedReflections::temporal_accumulation(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Temporal Accumulation", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_temporal_accumulation.current_output_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_temporal_accumulation.current_moments_image[m_common_resources->ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_temporal_accumulation.pipeline->handle());

    TemporalAccumulationPushConstants push_constants;

    push_constants.camera_delta  = m_common_resources->camera_delta;
    push_constants.frame_time    = m_common_resources->frame_time;
    push_constants.alpha         = m_temporal_accumulation.alpha;
    push_constants.moments_alpha = m_temporal_accumulation.moments_alpha;
    push_constants.g_buffer_mip  = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_temporal_accumulation.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_temporal_accumulation.current_write_ds[m_common_resources->ping_pong]->handle(),
        m_g_buffer->output_ds()->handle(),
        m_g_buffer->history_ds()->handle(),
        m_ray_trace.read_ds->handle(),
        m_temporal_accumulation.prev_read_ds[!m_common_resources->ping_pong]->handle(),
        m_common_resources->per_frame_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_temporal_accumulation.pipeline_layout->handle(), 0, 6, descriptor_sets, 1, &dynamic_offset);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_height) / float(NUM_THREADS))), 1);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_temporal_accumulation.current_output_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_temporal_accumulation.current_moments_image[m_common_resources->ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void RayTracedReflections::a_trous_filter(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("SVGF A-Trous Filter", cmd_buf);

    const uint32_t NUM_THREADS = 32;

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous.pipeline->handle());

    bool    ping_pong = false;
    int32_t read_idx  = 0;
    int32_t write_idx = 1;

    for (int i = 0; i < m_a_trous.filter_iterations; i++)
    {
        read_idx  = (int32_t)ping_pong;
        write_idx = (int32_t)!ping_pong;

        if (i == 0)
        {
            std::vector<VkMemoryBarrier> memory_barriers = {
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
            };

            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous.image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        else
        {
            std::vector<VkMemoryBarrier> memory_barriers = {
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
            };

            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous.image[read_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
                image_memory_barrier(m_a_trous.image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        ATrousFilterPushConstants push_constants;

        push_constants.radius       = m_a_trous.radius;
        push_constants.step_size    = 1 << i;
        push_constants.phi_color    = m_a_trous.phi_color;
        push_constants.phi_normal   = m_a_trous.phi_normal;
        push_constants.g_buffer_mip = m_g_buffer_mip;

        vkCmdPushConstants(cmd_buf->handle(), m_a_trous.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_a_trous.write_ds[write_idx]->handle(),
            i == 0 ? m_temporal_accumulation.output_only_read_ds->handle() : m_a_trous.read_ds[read_idx]->handle(),
            m_g_buffer->history_ds()->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous.pipeline_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

        vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_height) / float(NUM_THREADS))), 1);

        ping_pong = !ping_pong;

        if (m_a_trous.feedback_iteration == i)
        {
            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_a_trous.image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_temporal_accumulation.prev_image->handle(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresource_range);

            VkImageCopy image_copy_region {};
            image_copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_copy_region.srcSubresource.layerCount = 1;
            image_copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_copy_region.dstSubresource.layerCount = 1;
            image_copy_region.extent.width              = m_width;
            image_copy_region.extent.height             = m_height;
            image_copy_region.extent.depth              = 1;

            // Issue the copy command
            vkCmdCopyImage(
                cmd_buf->handle(),
                m_a_trous.image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_temporal_accumulation.prev_image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &image_copy_region);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_a_trous.image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_temporal_accumulation.prev_image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }
    }

    m_a_trous.read_idx = write_idx;

    std::vector<VkMemoryBarrier> memory_barriers = {
        memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    std::vector<VkImageMemoryBarrier> image_barriers = {
        image_memory_barrier(m_a_trous.image[write_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}