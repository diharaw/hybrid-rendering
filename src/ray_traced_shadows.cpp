#include "ray_traced_shadows.h"
#include "g_buffer.h"
#include "utilities.h"
#include <profiler.h>
#include <macros.h>
#include <imgui.h>

// -----------------------------------------------------------------------------------------------------------------------------------

static const int RAY_TRACE_NUM_THREADS_X = 8;
static const int RAY_TRACE_NUM_THREADS_Y = 4;

static const uint32_t TEMPORAL_ACCUMULATION_NUM_THREADS_X = 8;
static const uint32_t TEMPORAL_ACCUMULATION_NUM_THREADS_Y = 8;

// -----------------------------------------------------------------------------------------------------------------------------------

struct RayTracePushConstants
{
    float    bias;
    uint32_t num_frames;
    int32_t  g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct TemporalAccumulationPushConstants
{
    float   alpha;
    float   moments_alpha;
    int32_t g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct ATrousFilterPushConstants
{
    int     radius;
    int     step_size;
    float   phi_visibility;
    float   phi_normal;
    float   sigma_depth;
    int32_t g_buffer_mip;
    float   power;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct UpsamplePushConstants
{
    int32_t g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

const RayTracedShadows::OutputType RayTracedShadows::kOutputTypeEnums[] = {
    RayTracedShadows::OUTPUT_RAY_TRACE,
    RayTracedShadows::OUTPUT_TEMPORAL_ACCUMULATION,
    RayTracedShadows::OUTPUT_ATROUS,
    RayTracedShadows::OUTPUT_UPSAMPLE
};

// -----------------------------------------------------------------------------------------------------------------------------------

const std::string RayTracedShadows::kOutputTypeNames[] = {
    "Ray Trace",
    "Temporal Accumulation",
    "A-Trous",
    "Upsample"
};

// -----------------------------------------------------------------------------------------------------------------------------------

RayTracedShadows::RayTracedShadows(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, RayTraceScale scale) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer), m_scale(scale)
{
    auto vk_backend = m_backend.lock();

    float scale_divisor = powf(2.0f, float(scale));

    m_width  = vk_backend->swap_chain_extents().width / scale_divisor;
    m_height = vk_backend->swap_chain_extents().height / scale_divisor;

    m_g_buffer_mip = static_cast<uint32_t>(scale);

    create_images();
    create_buffers();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipelines();
}

// -----------------------------------------------------------------------------------------------------------------------------------

RayTracedShadows::~RayTracedShadows()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Traced Shadows", cmd_buf);

    clear_images(cmd_buf);
    ray_trace(cmd_buf);

    if (m_denoise)
    {
        reset_args(cmd_buf);
        temporal_accumulation(cmd_buf);
        a_trous_filter(cmd_buf);

        if (m_scale != RAY_TRACE_SCALE_FULL_RES)
            upsample(cmd_buf);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::gui()
{
    ImGui::Checkbox("Denoise", &m_denoise);
    ImGui::InputFloat("Bias", &m_ray_trace.bias);
    ImGui::InputFloat("Alpha", &m_temporal_accumulation.alpha);
    ImGui::InputFloat("Alpha Moments", &m_temporal_accumulation.moments_alpha);
    ImGui::InputFloat("Phi Visibility", &m_a_trous.phi_visibility);
    ImGui::InputFloat("Phi Normal", &m_a_trous.phi_normal);
    ImGui::InputFloat("Sigma Depth", &m_a_trous.sigma_depth);
    ImGui::SliderInt("Filter Iterations", &m_a_trous.filter_iterations, 1, 5);
    ImGui::SliderFloat("Power", &m_a_trous.power, 1.0f, 50.0f);
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::DescriptorSet::Ptr RayTracedShadows::output_ds()
{
    if (m_denoise)
    {
        if (m_current_output == OUTPUT_RAY_TRACE)
            return m_ray_trace.read_ds;
        else if (m_current_output == OUTPUT_TEMPORAL_ACCUMULATION)
            return m_temporal_accumulation.output_only_read_ds;
        else if (m_current_output == OUTPUT_ATROUS)
            return m_a_trous.read_ds[m_a_trous.read_idx];
        else
        {
            if (m_scale == RAY_TRACE_SCALE_FULL_RES)
                return m_a_trous.read_ds[m_a_trous.read_idx];
            else
                return m_upsample.read_ds;
        }
    }
    else
        return m_ray_trace.read_ds;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::create_images()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        m_ray_trace.image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, static_cast<uint32_t>(ceil(float(m_width) / float(RAY_TRACE_NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_height) / float(RAY_TRACE_NUM_THREADS_Y))), 1, 1, 1, VK_FORMAT_R32_UINT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_ray_trace.image->set_name("Shadows Ray Trace");

        m_ray_trace.view = dw::vk::ImageView::create(backend, m_ray_trace.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_ray_trace.view->set_name("Shadows Ray Trace");
    }

    // Reprojection
    {
        m_temporal_accumulation.current_output_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_temporal_accumulation.current_output_image->set_name("Shadows Reprojection Output");

        m_temporal_accumulation.current_output_view = dw::vk::ImageView::create(backend, m_temporal_accumulation.current_output_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_temporal_accumulation.current_output_view->set_name("Shadows Reprojection Output");

        for (int i = 0; i < 2; i++)
        {
            m_temporal_accumulation.current_moments_image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
            m_temporal_accumulation.current_moments_image[i]->set_name("Shadows Reprojection Moments " + std::to_string(i));

            m_temporal_accumulation.current_moments_view[i] = dw::vk::ImageView::create(backend, m_temporal_accumulation.current_moments_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            m_temporal_accumulation.current_moments_view[i]->set_name("Shadows Reprojection Moments " + std::to_string(i));
        }

        m_temporal_accumulation.prev_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_temporal_accumulation.prev_image->set_name("Shadows Previous Reprojection");

        m_temporal_accumulation.prev_view = dw::vk::ImageView::create(backend, m_temporal_accumulation.prev_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_temporal_accumulation.prev_view->set_name("Shadows Previous Reprojection");
    }

    // A-Trous Filter
    for (int i = 0; i < 2; i++)
    {
        m_a_trous.image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_a_trous.image[i]->set_name("A-Trous Filter " + std::to_string(i));

        m_a_trous.view[i] = dw::vk::ImageView::create(backend, m_a_trous.image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_a_trous.view[i]->set_name("A-Trous Filter View " + std::to_string(i));
    }

    // Upsample
    {
        auto vk_backend = m_backend.lock();

        m_upsample.image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, vk_backend->swap_chain_extents().width, vk_backend->swap_chain_extents().height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_upsample.image->set_name("Shadows Upsample");

        m_upsample.image_view = dw::vk::ImageView::create(backend, m_upsample.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_upsample.image_view->set_name("Shadows Upsample");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::create_buffers()
{
    auto backend = m_backend.lock();

    m_temporal_accumulation.denoise_tile_coords_buffer   = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(glm::ivec2) * static_cast<uint32_t>(ceil(float(m_width) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_X))) * static_cast<uint32_t>(ceil(float(m_height) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_Y))), VMA_MEMORY_USAGE_GPU_ONLY, 0);
    m_temporal_accumulation.denoise_dispatch_args_buffer = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, sizeof(int32_t) * 3, VMA_MEMORY_USAGE_GPU_ONLY, 0);

    m_temporal_accumulation.shadow_tile_coords_buffer   = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(glm::ivec2) * static_cast<uint32_t>(ceil(float(m_width) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_X))) * static_cast<uint32_t>(ceil(float(m_height) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_Y))), VMA_MEMORY_USAGE_GPU_ONLY, 0);
    m_temporal_accumulation.shadow_dispatch_args_buffer = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, sizeof(int32_t) * 3, VMA_MEMORY_USAGE_GPU_ONLY, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::create_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        m_ray_trace.write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_ray_trace.write_ds->set_name("Shadows Ray Trace Write");

        m_ray_trace.read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_ray_trace.read_ds->set_name("Shadows Ray Trace Read");
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
        m_temporal_accumulation.current_write_ds[i]->set_name("Temporal Accumulation Write " + std::to_string(i));

        m_temporal_accumulation.current_read_ds[i] = backend->allocate_descriptor_set(m_temporal_accumulation.read_ds_layout);
        m_temporal_accumulation.current_read_ds[i]->set_name("Temporal Accumulation Read " + std::to_string(i));

        m_temporal_accumulation.prev_read_ds[i] = backend->allocate_descriptor_set(m_temporal_accumulation.read_ds_layout);
        m_temporal_accumulation.prev_read_ds[i]->set_name("Temporal Accumulation Prev Read " + std::to_string(i));
    }

    m_temporal_accumulation.output_only_read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
    m_temporal_accumulation.output_only_read_ds->set_name("Temporal Accumulation Output Only Read");

    // Indirect Buffer
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_temporal_accumulation.indirect_buffer_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);

        m_temporal_accumulation.indirect_buffer_ds = backend->allocate_descriptor_set(m_temporal_accumulation.indirect_buffer_ds_layout);
        m_temporal_accumulation.indirect_buffer_ds->set_name("Temporal Accumulation Indirect Buffer");
    }

    // A-Trous
    for (int i = 0; i < 2; i++)
    {
        m_a_trous.read_ds[i] = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_a_trous.read_ds[i]->set_name("A-Trous Read " + std::to_string(i));

        m_a_trous.write_ds[i] = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_a_trous.write_ds[i]->set_name("A-Trous Write " + std::to_string(i));
    }

    // Upsample
    {
        m_upsample.write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_upsample.write_ds->set_name("Shadows Upsample Write");

        m_upsample.read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_upsample.read_ds->set_name("Shadows Upsample Read");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::write_descriptor_sets()
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

    // Indirect Buffer
    {
        std::vector<VkDescriptorBufferInfo> buffer_infos;
        std::vector<VkWriteDescriptorSet>   write_datas;
        VkWriteDescriptorSet                write_data;

        buffer_infos.reserve(4);
        write_datas.reserve(4);

        {
            VkDescriptorBufferInfo buffer_info;

            buffer_info.range  = m_temporal_accumulation.denoise_tile_coords_buffer->size();
            buffer_info.offset = 0;
            buffer_info.buffer = m_temporal_accumulation.denoise_tile_coords_buffer->handle();

            buffer_infos.push_back(buffer_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write_data.pBufferInfo     = &buffer_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_temporal_accumulation.indirect_buffer_ds->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorBufferInfo buffer_info;

            buffer_info.range  = m_temporal_accumulation.denoise_dispatch_args_buffer->size();
            buffer_info.offset = 0;
            buffer_info.buffer = m_temporal_accumulation.denoise_dispatch_args_buffer->handle();

            buffer_infos.push_back(buffer_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write_data.pBufferInfo     = &buffer_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_temporal_accumulation.indirect_buffer_ds->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorBufferInfo buffer_info;

            buffer_info.range  = m_temporal_accumulation.shadow_tile_coords_buffer->size();
            buffer_info.offset = 0;
            buffer_info.buffer = m_temporal_accumulation.shadow_tile_coords_buffer->handle();

            buffer_infos.push_back(buffer_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write_data.pBufferInfo     = &buffer_infos.back();
            write_data.dstBinding      = 2;
            write_data.dstSet          = m_temporal_accumulation.indirect_buffer_ds->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorBufferInfo buffer_info;

            buffer_info.range  = m_temporal_accumulation.shadow_dispatch_args_buffer->size();
            buffer_info.offset = 0;
            buffer_info.buffer = m_temporal_accumulation.shadow_dispatch_args_buffer->handle();

            buffer_infos.push_back(buffer_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write_data.pBufferInfo     = &buffer_infos.back();
            write_data.dstBinding      = 3;
            write_data.dstSet          = m_temporal_accumulation.indirect_buffer_ds->handle();

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

    // Upsample
    {
        // write
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_upsample.image_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_upsample.write_ds->handle();

            vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
        }

        // read
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_upsample.image_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_upsample.read_ds->handle();

            vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::create_pipelines()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        dw::vk::ShaderModule::Ptr shader_module = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadows_ray_trace.comp.spv");

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_common_resources->current_scene()->descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        pl_desc.add_descriptor_set_layout(m_common_resources->blue_noise_ds_layout);

        pl_desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayTracePushConstants));

        m_ray_trace.pipeline_layout = dw::vk::PipelineLayout::create(backend, pl_desc);
        m_ray_trace.pipeline_layout->set_name("Ray Trace Pipeline Layout");

        dw::vk::ComputePipeline::Desc desc;

        desc.set_shader_stage(shader_module, "main");
        desc.set_pipeline_layout(m_ray_trace.pipeline_layout);

        m_ray_trace.pipeline = dw::vk::ComputePipeline::create(backend, desc);
    }

    // Reset Args
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_temporal_accumulation.indirect_buffer_ds_layout);

        m_reset_args.pipeline_layout = dw::vk::PipelineLayout::create(backend, desc);
        m_reset_args.pipeline_layout->set_name("Reset Args Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadows_denoise_reset_args.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_reset_args.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_reset_args.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
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
        desc.add_descriptor_set_layout(m_temporal_accumulation.indirect_buffer_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TemporalAccumulationPushConstants));

        m_temporal_accumulation.pipeline_layout = dw::vk::PipelineLayout::create(backend, desc);
        m_temporal_accumulation.pipeline_layout->set_name("Reprojection Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadows_denoise_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_temporal_accumulation.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_temporal_accumulation.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // Copy Shadow Tiles
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_temporal_accumulation.indirect_buffer_ds_layout);

        m_copy_shadow_tiles.pipeline_layout = dw::vk::PipelineLayout::create(backend, desc);
        m_copy_shadow_tiles.pipeline_layout->set_name("Copy Shadow Tiles Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadows_denoise_copy_shadow_tiles.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_copy_shadow_tiles.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_copy_shadow_tiles.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // A-Trous Filter
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_temporal_accumulation.indirect_buffer_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ATrousFilterPushConstants));

        m_a_trous.pipeline_layout = dw::vk::PipelineLayout::create(backend, desc);
        m_a_trous.pipeline_layout->set_name("A-Trous Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadows_denoise_atrous.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_a_trous.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_a_trous.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // Upsample
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpsamplePushConstants));

        m_upsample.layout = dw::vk::PipelineLayout::create(backend, desc);
        m_upsample.layout->set_name("Shadows Upsample Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/shadows_upsample.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_upsample.layout);
        comp_desc.set_shader_stage(module, "main");

        m_upsample.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
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

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf)
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

    pipeline_barrier(cmd_buf, memory_barriers, image_barriers, {}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_ray_trace.pipeline->handle());

    RayTracePushConstants push_constants;

    push_constants.bias         = m_ray_trace.bias;
    push_constants.num_frames   = m_common_resources->num_frames;
    push_constants.g_buffer_mip = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_ray_trace.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene()->descriptor_set()->handle(),
        m_ray_trace.write_ds->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_g_buffer->output_ds()->handle(),
        m_common_resources->blue_noise_ds[BLUE_NOISE_1SPP]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_ray_trace.pipeline_layout->handle(), 0, 5, descriptor_sets, 1, &dynamic_offset);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_width) / float(RAY_TRACE_NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_height) / float(RAY_TRACE_NUM_THREADS_Y))), 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_ray_trace.image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::reset_args(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Reset Args", cmd_buf);

    {
        std::vector<VkBufferMemoryBarrier> buffer_barriers = {
            buffer_memory_barrier(m_temporal_accumulation.denoise_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.denoise_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.shadow_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.shadow_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, {}, {}, buffer_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reset_args.pipeline->handle());

    VkDescriptorSet descriptor_sets[] = {
        m_temporal_accumulation.indirect_buffer_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reset_args.pipeline_layout->handle(), 0, 1, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), 1, 1, 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::temporal_accumulation(dw::vk::CommandBuffer::Ptr cmd_buf)
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

        std::vector<VkBufferMemoryBarrier> buffer_barriers = {
            buffer_memory_barrier(m_temporal_accumulation.denoise_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.denoise_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.shadow_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.shadow_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, buffer_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_temporal_accumulation.pipeline->handle());

    TemporalAccumulationPushConstants push_constants;

    push_constants.alpha         = m_temporal_accumulation.alpha;
    push_constants.moments_alpha = m_temporal_accumulation.moments_alpha;
    push_constants.g_buffer_mip  = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_temporal_accumulation.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_temporal_accumulation.current_write_ds[m_common_resources->ping_pong]->handle(),
        m_g_buffer->output_ds()->handle(),
        m_g_buffer->history_ds()->handle(),
        m_ray_trace.read_ds->handle(),
        m_temporal_accumulation.prev_read_ds[!m_common_resources->ping_pong]->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_temporal_accumulation.indirect_buffer_ds->handle()
    };

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_temporal_accumulation.pipeline_layout->handle(), 0, 7, descriptor_sets, 1, &dynamic_offset);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_width) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_height) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_Y))), 1);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_temporal_accumulation.current_output_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_temporal_accumulation.current_moments_image[m_common_resources->ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkBufferMemoryBarrier> buffer_barriers = {
            buffer_memory_barrier(m_temporal_accumulation.denoise_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            buffer_memory_barrier(m_temporal_accumulation.denoise_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
            buffer_memory_barrier(m_temporal_accumulation.shadow_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            buffer_memory_barrier(m_temporal_accumulation.shadow_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, buffer_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::a_trous_filter(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("A-Trous Filter", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    bool    ping_pong = false;
    int32_t read_idx  = 0;
    int32_t write_idx = 1;

    for (int i = 0; i < m_a_trous.filter_iterations; i++)
    {
        read_idx  = (int32_t)ping_pong;
        write_idx = (int32_t)!ping_pong;

        if (i == 0)
        {
            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous.image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        else
        {
            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous.image[read_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
                image_memory_barrier(m_a_trous.image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        {
            VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkClearColorValue color;

            color.float32[0] = 1.0f;
            color.float32[1] = 1.0f;
            color.float32[2] = 1.0f;
            color.float32[3] = 1.0f;

            vkCmdClearColorImage(cmd_buf->handle(), m_a_trous.image[write_idx]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        }

        {
            DW_SCOPED_SAMPLE("Copy Shadow Tiles", cmd_buf);

            vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_copy_shadow_tiles.pipeline->handle());

            VkDescriptorSet descriptor_sets[] = {
                m_a_trous.write_ds[write_idx]->handle(),
                m_temporal_accumulation.indirect_buffer_ds->handle()
            };

            vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_copy_shadow_tiles.pipeline_layout->handle(), 0, 2, descriptor_sets, 0, nullptr);

            vkCmdDispatchIndirect(cmd_buf->handle(), m_temporal_accumulation.shadow_dispatch_args_buffer->handle(), 0);
        }

        {
            DW_SCOPED_SAMPLE("Iteration " + std::to_string(i), cmd_buf);

            vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous.pipeline->handle());

            ATrousFilterPushConstants push_constants;

            push_constants.radius         = m_a_trous.radius;
            push_constants.step_size      = 1 << i;
            push_constants.phi_visibility = m_a_trous.phi_visibility;
            push_constants.phi_normal     = m_a_trous.phi_normal;
            push_constants.sigma_depth    = m_a_trous.sigma_depth;
            push_constants.g_buffer_mip   = m_g_buffer_mip;
            push_constants.power          = i == (m_a_trous.filter_iterations - 1) ? m_a_trous.power : 0.0f;

            vkCmdPushConstants(cmd_buf->handle(), m_a_trous.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

            VkDescriptorSet descriptor_sets[] = {
                m_a_trous.write_ds[write_idx]->handle(),
                i == 0 ? m_temporal_accumulation.output_only_read_ds->handle() : m_a_trous.read_ds[read_idx]->handle(),
                m_g_buffer->output_ds()->handle(),
                m_temporal_accumulation.indirect_buffer_ds->handle()
            };

            vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous.pipeline_layout->handle(), 0, 4, descriptor_sets, 0, nullptr);

            vkCmdDispatchIndirect(cmd_buf->handle(), m_temporal_accumulation.denoise_dispatch_args_buffer->handle(), 0);
        }

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

    pipeline_barrier(cmd_buf, memory_barriers, image_barriers, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedShadows::upsample(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Upsample", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_upsample.image->handle(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        subresource_range);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_upsample.pipeline->handle());

    UpsamplePushConstants push_constants;

    push_constants.g_buffer_mip = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_upsample.layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_upsample.write_ds->handle(),
        m_a_trous.read_ds[m_a_trous.read_idx]->handle(),
        m_g_buffer->output_ds()->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_upsample.layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

    const int NUM_THREADS_X = 32;
    const int NUM_THREADS_Y = 32;

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_upsample.image->width()) / float(NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_upsample.image->height()) / float(NUM_THREADS_Y))), 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_upsample.image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

// -----------------------------------------------------------------------------------------------------------------------------------