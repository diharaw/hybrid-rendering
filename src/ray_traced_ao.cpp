#include "ray_traced_ao.h"
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
    uint32_t num_frames;
    float    ray_length;
    float    bias;
    int32_t  g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct TemporalReprojectionPushConstants
{
    float   alpha;
    int32_t g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct DisocclusionBlurPushConstants
{
    glm::vec4 z_buffer_params;
    float     threshold;
    int32_t   blur_radius;
    uint32_t  enabled;
    int32_t   g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct BilateralBlurPushConstants
{
    glm::vec4  z_buffer_params;
    glm::ivec2 direction;
    int32_t    radius;
    int32_t    g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct UpsamplePushConstants
{
    int32_t g_buffer_mip;
    float   power;
};

// -----------------------------------------------------------------------------------------------------------------------------------

const RayTracedAO::OutputType RayTracedAO::kOutputTypeEnums[] = {
    RayTracedAO::OUTPUT_RAY_TRACE,
    RayTracedAO::OUTPUT_TEMPORAL_ACCUMULATION,
    RayTracedAO::OUTPUT_BILATERAL_BLUR,
    RayTracedAO::OUTPUT_DISOCCLUSION_BLUR,
    RayTracedAO::OUTPUT_UPSAMPLE
};

// -----------------------------------------------------------------------------------------------------------------------------------

const std::string RayTracedAO::kOutputTypeNames[] = {
    "Ray Trace",
    "Temporal Accumulation",
    "Bilateral Blur",
    "Disocclusion Blur",
    "Upsample"
};

// -----------------------------------------------------------------------------------------------------------------------------------

RayTracedAO::RayTracedAO(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, RayTraceScale scale) :
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

    clear_images(cmd_buf);
    ray_trace(cmd_buf);

    if (m_denoise)
    {
        denoise(cmd_buf);

        if (m_scale != RAY_TRACE_SCALE_FULL_RES)
            upsample(cmd_buf);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::gui()
{
    ImGui::Checkbox("Denoise", &m_denoise);
    ImGui::Checkbox("Disocclusion Blur", &m_disocclusion_blur.enabled);
    ImGui::SliderFloat("Ray Length", &m_ray_trace.ray_length, 1.0f, 100.0f);
    ImGui::SliderFloat("Power", &m_upsample.power, 1.0f, 5.0f);
    ImGui::InputFloat("Bias", &m_ray_trace.bias);
    ImGui::SliderFloat("Temporal Alpha", &m_temporal_accumulation.alpha, 0.0f, 0.5f);
    ImGui::SliderInt("Blur Radius", &m_bilateral_blur.blur_radius, 1, 10);
    ImGui::SliderInt("Disocclusion Blur Radius", &m_disocclusion_blur.blur_radius, 1, 20);
    ImGui::SliderInt("Disocclusion Blur Threshold", &m_disocclusion_blur.threshold, 1, 15);
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::DescriptorSet::Ptr RayTracedAO::output_ds()
{
    if (m_denoise)
    {
        if (m_current_output == OUTPUT_RAY_TRACE)
            return m_ray_trace.read_ds;
        else if (m_current_output == OUTPUT_TEMPORAL_ACCUMULATION)
            return m_temporal_accumulation.output_read_ds[m_common_resources->ping_pong];
        else if (m_current_output == OUTPUT_BILATERAL_BLUR)
            return m_bilateral_blur.read_ds[1];
        else if (m_current_output == OUTPUT_DISOCCLUSION_BLUR)
            return m_disocclusion_blur.read_ds;
        else
        {
            if (m_scale == RAY_TRACE_SCALE_FULL_RES)
                return m_disocclusion_blur.read_ds;
            else
                return m_upsample.read_ds;
        }
    }
    else
        return m_ray_trace.read_ds;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::create_images()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        m_ray_trace.image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, static_cast<uint32_t>(ceil(float(m_width) / float(RAY_TRACE_NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_height) / float(RAY_TRACE_NUM_THREADS_Y))), 1, 1, 1, VK_FORMAT_R32_UINT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_ray_trace.image->set_name("AO Ray Trace");

        m_ray_trace.view = dw::vk::ImageView::create(backend, m_ray_trace.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_ray_trace.view->set_name("AO Ray Trace");
    }

    // Temporal Reprojection
    for (int i = 0; i < 2; i++)
    {
        m_temporal_accumulation.color_image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_temporal_accumulation.color_image[i]->set_name("AO Denoise Reprojection " + std::to_string(i));

        m_temporal_accumulation.color_view[i] = dw::vk::ImageView::create(backend, m_temporal_accumulation.color_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_temporal_accumulation.color_view[i]->set_name("AO Denoise Reprojection " + std::to_string(i));

        m_temporal_accumulation.history_length_image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_temporal_accumulation.history_length_image[i]->set_name("AO Denoise Reprojection History " + std::to_string(i));

        m_temporal_accumulation.history_length_view[i] = dw::vk::ImageView::create(backend, m_temporal_accumulation.history_length_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_temporal_accumulation.history_length_view[i]->set_name("AO Denoise Reprojection History " + std::to_string(i));
    }

    // Disocclusion Blur
    {
        m_disocclusion_blur.image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_disocclusion_blur.image->set_name("AO Disocclusion Blur");

        m_disocclusion_blur.image_view = dw::vk::ImageView::create(backend, m_disocclusion_blur.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_disocclusion_blur.image_view->set_name("AO Disocclusion Blur");
    }

    // Bilateral Blur
    for (int i = 0; i < 2; i++)
    {
        m_bilateral_blur.image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_bilateral_blur.image[i]->set_name("AO Denoise Blur " + std::to_string(i));

        m_bilateral_blur.image_view[i] = dw::vk::ImageView::create(backend, m_bilateral_blur.image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_bilateral_blur.image_view[i]->set_name("AO Denoise Blur " + std::to_string(i));
    }

    // Upsample
    {
        auto vk_backend = m_backend.lock();

        m_upsample.image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, vk_backend->swap_chain_extents().width, vk_backend->swap_chain_extents().height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_upsample.image->set_name("AO Upsample");

        m_upsample.image_view = dw::vk::ImageView::create(backend, m_upsample.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_upsample.image_view->set_name("AO Upsample");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::create_buffers()
{
    auto backend = m_backend.lock();

    m_temporal_accumulation.denoise_tile_coords_buffer   = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(glm::ivec2) * static_cast<uint32_t>(ceil(float(m_width) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_X))) * static_cast<uint32_t>(ceil(float(m_height) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_Y))), VMA_MEMORY_USAGE_GPU_ONLY, 0);
    m_temporal_accumulation.denoise_dispatch_args_buffer = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, sizeof(int32_t) * 3, VMA_MEMORY_USAGE_GPU_ONLY, 0);

    m_temporal_accumulation.disocclusion_tile_coords_buffer   = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(glm::ivec2) * static_cast<uint32_t>(ceil(float(m_width) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_X))) * static_cast<uint32_t>(ceil(float(m_height) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_Y))), VMA_MEMORY_USAGE_GPU_ONLY, 0);
    m_temporal_accumulation.disocclusion_dispatch_args_buffer = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, sizeof(int32_t) * 3, VMA_MEMORY_USAGE_GPU_ONLY, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::create_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        m_ray_trace.write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_ray_trace.write_ds->set_name("AO Ray Trace Write");

        m_ray_trace.read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_ray_trace.read_ds->set_name("AO Ray Trace Read");

        m_ray_trace.bilinear_read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_ray_trace.bilinear_read_ds->set_name("AO Ray Trace Bilinear Output Read");
    }

    // Temporal Reprojection
    {
        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

            m_temporal_accumulation.write_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
            m_temporal_accumulation.write_ds_layout->set_name("AO Reprojection Write DS Layout");
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

            m_temporal_accumulation.read_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
            m_temporal_accumulation.read_ds_layout->set_name("AO Reprojection Read DS Layout");
        }

        for (int i = 0; i < 2; i++)
        {
            m_temporal_accumulation.write_ds[i] = backend->allocate_descriptor_set(m_temporal_accumulation.write_ds_layout);
            m_temporal_accumulation.write_ds[i]->set_name("AO Reprojection Write " + std::to_string(i));

            m_temporal_accumulation.read_ds[i] = backend->allocate_descriptor_set(m_temporal_accumulation.read_ds_layout);
            m_temporal_accumulation.read_ds[i]->set_name("AO Reprojection Read " + std::to_string(i));

            m_temporal_accumulation.output_read_ds[i] = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
            m_temporal_accumulation.output_read_ds[i]->set_name("AO Reprojection Output Read " + std::to_string(i));
        }
    }

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

    // Disocclusion Blur
    {
        m_disocclusion_blur.write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_disocclusion_blur.write_ds->set_name("AO Disocclusion Blur Write");

        m_disocclusion_blur.read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_disocclusion_blur.read_ds->set_name("AO Disocclusion Blur Read");
    }

    // Bilateral Blur
    {
        for (int i = 0; i < 2; i++)
        {
            m_bilateral_blur.write_ds[i] = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
            m_bilateral_blur.write_ds[i]->set_name("AO Blur Write " + std::to_string(i));

            m_bilateral_blur.read_ds[i] = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
            m_bilateral_blur.read_ds[i]->set_name("AO Blur Read " + std::to_string(i));
        }
    }

    // Upsample
    {
        m_upsample.write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_upsample.write_ds->set_name("AO Upsample Write");

        m_upsample.read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_upsample.read_ds->set_name("AO Upsample Read");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::write_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Ray Trace
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

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->bilinear_sampler()->handle();
            sampler_image_info.imageView   = m_ray_trace.view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_ray_trace.bilinear_read_ds->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Temporal Reprojection
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(4);
        write_datas.reserve(4);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_temporal_accumulation.color_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_temporal_accumulation.write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_temporal_accumulation.history_length_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_temporal_accumulation.write_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(4);
        write_datas.reserve(4);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_temporal_accumulation.color_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_temporal_accumulation.read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_temporal_accumulation.history_length_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_temporal_accumulation.read_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_temporal_accumulation.color_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_temporal_accumulation.output_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }
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

            buffer_info.range  = m_temporal_accumulation.disocclusion_tile_coords_buffer->size();
            buffer_info.offset = 0;
            buffer_info.buffer = m_temporal_accumulation.disocclusion_tile_coords_buffer->handle();

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

            buffer_info.range  = m_temporal_accumulation.disocclusion_dispatch_args_buffer->size();
            buffer_info.offset = 0;
            buffer_info.buffer = m_temporal_accumulation.disocclusion_dispatch_args_buffer->handle();

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

    // Disocclusion Blur
    {
        // write
        {
            VkDescriptorImageInfo storage_image_info;

    storage_image_info.sampler     = VK_NULL_HANDLE;
    storage_image_info.imageView   = m_disocclusion_blur.image_view->handle();
    storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write_data;

    DW_ZERO_MEMORY(write_data);

    write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_data.descriptorCount = 1;
    write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write_data.pImageInfo      = &storage_image_info;
    write_data.dstBinding      = 0;
    write_data.dstSet          = m_disocclusion_blur.write_ds->handle();

    vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
}

// read
{
    VkDescriptorImageInfo sampler_image_info;

    sampler_image_info.sampler     = backend->nearest_sampler()->handle();
    sampler_image_info.imageView   = m_disocclusion_blur.image_view->handle();
    sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write_data;

    DW_ZERO_MEMORY(write_data);

    write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_data.descriptorCount = 1;
    write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write_data.pImageInfo      = &sampler_image_info;
    write_data.dstBinding      = 0;
    write_data.dstSet          = m_disocclusion_blur.read_ds->handle();

    vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
}
}

// Bilateral Blur
{
    for (int i = 0; i < 2; i++)
    {
        // write
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_bilateral_blur.image_view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_bilateral_blur.write_ds[i]->handle();

            vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
        }

        // read
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_bilateral_blur.image_view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_bilateral_blur.read_ds[i]->handle();

            vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
        }
    }
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

void RayTracedAO::create_pipeline()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        dw::vk::ShaderModule::Ptr shader_module = dw::vk::ShaderModule::create_from_file(backend, "shaders/ao_ray_trace.comp.spv");

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_common_resources->current_scene()->descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        pl_desc.add_descriptor_set_layout(m_common_resources->blue_noise_ds_layout);

        pl_desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayTracePushConstants));

        m_ray_trace.pipeline_layout = dw::vk::PipelineLayout::create(backend, pl_desc);
        m_ray_trace.pipeline_layout->set_name("AO Ray Trace Pipeline Layout");

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

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/ao_denoise_reset_args.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_reset_args.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_reset_args.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // Temporal Reprojection
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_temporal_accumulation.write_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_temporal_accumulation.read_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
        desc.add_descriptor_set_layout(m_temporal_accumulation.indirect_buffer_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TemporalReprojectionPushConstants));

        m_temporal_accumulation.pipeline_layout = dw::vk::PipelineLayout::create(backend, desc);
        m_temporal_accumulation.pipeline_layout->set_name("AO Reprojection Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/ao_denoise_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_temporal_accumulation.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_temporal_accumulation.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // Disocclusion Blur
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_temporal_accumulation.read_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_temporal_accumulation.indirect_buffer_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DisocclusionBlurPushConstants));

        m_disocclusion_blur.layout = dw::vk::PipelineLayout::create(backend, desc);
        m_disocclusion_blur.layout->set_name("AO Disocclusion Blur Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/ao_denoise_disocclusion_blur.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_disocclusion_blur.layout);
        comp_desc.set_shader_stage(module, "main");

        m_disocclusion_blur.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // Bilateral Blur
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_temporal_accumulation.read_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_temporal_accumulation.indirect_buffer_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BilateralBlurPushConstants));

        m_bilateral_blur.layout = dw::vk::PipelineLayout::create(backend, desc);
        m_bilateral_blur.layout->set_name("AO Blur Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/ao_denoise_bilateral_blur.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_bilateral_blur.layout);
        comp_desc.set_shader_stage(module, "main");

        m_bilateral_blur.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }

    // Upsample
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpsamplePushConstants));

        m_upsample.layout = dw::vk::PipelineLayout::create(backend, desc);
        m_upsample.layout->set_name("AO Upsample Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(backend, "shaders/ao_upsample.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_upsample.layout);
        comp_desc.set_shader_stage(module, "main");

        m_upsample.pipeline = dw::vk::ComputePipeline::create(backend, comp_desc);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
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
            m_temporal_accumulation.history_length_image[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temporal_accumulation.color_image[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdClearColorImage(cmd_buf->handle(), m_temporal_accumulation.history_length_image[!m_common_resources->ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_temporal_accumulation.color_image[!m_common_resources->ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temporal_accumulation.history_length_image[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temporal_accumulation.color_image[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        m_first_frame = false;
    }
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

    std::vector<VkImageMemoryBarrier> image_barriers = {
        image_memory_barrier(m_ray_trace.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
    };

    pipeline_barrier(cmd_buf, memory_barriers, image_barriers, {}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_ray_trace.pipeline->handle());

    RayTracePushConstants push_constants;

    push_constants.num_frames   = m_common_resources->num_frames;
    push_constants.ray_length   = m_ray_trace.ray_length;
    push_constants.bias         = m_ray_trace.bias;
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

void RayTracedAO::denoise(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Denoise", cmd_buf);

    reset_args(cmd_buf);
    temporal_accumulation(cmd_buf);
    bilateral_blur(cmd_buf);
    disocclusion_blur(cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::upsample(dw::vk::CommandBuffer::Ptr cmd_buf)
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
    push_constants.power        = m_upsample.power;

    vkCmdPushConstants(cmd_buf->handle(), m_upsample.layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_upsample.write_ds->handle(),
        m_disocclusion_blur.read_ds->handle(),
        m_g_buffer->output_ds()->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_upsample.layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

    const int NUM_THREADS_X = 8;
    const int NUM_THREADS_Y = 8;

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_upsample.image->width()) / float(NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_upsample.image->height()) / float(NUM_THREADS_Y))), 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_upsample.image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::reset_args(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Reset Args", cmd_buf);

    {
        std::vector<VkBufferMemoryBarrier> buffer_barriers = {
            buffer_memory_barrier(m_temporal_accumulation.denoise_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.denoise_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.disocclusion_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.disocclusion_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
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

void RayTracedAO::temporal_accumulation(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Temporal Accumulation", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_temporal_accumulation.color_image[m_common_resources->ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_temporal_accumulation.history_length_image[m_common_resources->ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkBufferMemoryBarrier> buffer_barriers = {
            buffer_memory_barrier(m_temporal_accumulation.denoise_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.denoise_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.disocclusion_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.disocclusion_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, buffer_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_temporal_accumulation.pipeline->handle());

    TemporalReprojectionPushConstants push_constants;

    push_constants.alpha        = m_temporal_accumulation.alpha;
    push_constants.g_buffer_mip = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_temporal_accumulation.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_temporal_accumulation.write_ds[m_common_resources->ping_pong]->handle(),
        m_g_buffer->output_ds()->handle(),
        m_g_buffer->history_ds()->handle(),
        m_ray_trace.read_ds->handle(),
        m_temporal_accumulation.output_read_ds[!m_common_resources->ping_pong]->handle(),
        m_temporal_accumulation.read_ds[!m_common_resources->ping_pong]->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_temporal_accumulation.indirect_buffer_ds->handle()
    };

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_temporal_accumulation.pipeline_layout->handle(), 0, 8, descriptor_sets, 1, &dynamic_offset);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_width) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_height) / float(TEMPORAL_ACCUMULATION_NUM_THREADS_Y))), 1);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_temporal_accumulation.color_image[m_common_resources->ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_temporal_accumulation.history_length_image[m_common_resources->ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkBufferMemoryBarrier> buffer_barriers = {
            buffer_memory_barrier(m_temporal_accumulation.denoise_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.denoise_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.disocclusion_tile_coords_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            buffer_memory_barrier(m_temporal_accumulation.disocclusion_dispatch_args_buffer, 0, VK_WHOLE_SIZE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, buffer_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::disocclusion_blur(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Disocclusion Blur", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_disocclusion_blur.pipeline->handle());

    DisocclusionBlurPushConstants push_constants;

    push_constants.z_buffer_params = m_common_resources->z_buffer_params;
    push_constants.enabled         = (uint32_t)m_disocclusion_blur.enabled;
    push_constants.blur_radius     = m_disocclusion_blur.blur_radius;
    push_constants.threshold       = (float)m_disocclusion_blur.threshold;
    push_constants.g_buffer_mip    = m_g_buffer_mip;

    vkCmdPushConstants(cmd_buf->handle(), m_disocclusion_blur.layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_disocclusion_blur.write_ds->handle(),
        m_bilateral_blur.read_ds[1]->handle(),
        m_temporal_accumulation.read_ds[m_common_resources->ping_pong]->handle(),
        m_g_buffer->output_ds()->handle(),
        m_temporal_accumulation.indirect_buffer_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_disocclusion_blur.layout->handle(), 0, 5, descriptor_sets, 0, nullptr);

    const int NUM_THREADS_X = 8;
    const int NUM_THREADS_Y = 8;

    vkCmdDispatchIndirect(cmd_buf->handle(), m_temporal_accumulation.disocclusion_dispatch_args_buffer->handle(), 0);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_disocclusion_blur.image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void RayTracedAO::bilateral_blur(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Bilateral Blur", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    // Vertical
    {
        DW_SCOPED_SAMPLE("Vertical", cmd_buf);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_bilateral_blur.image[0]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_disocclusion_blur.image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        {
            VkClearColorValue color;

            color.float32[0] = 1.0f;
            color.float32[1] = 1.0f;
            color.float32[2] = 1.0f;
            color.float32[3] = 1.0f;

            vkCmdClearColorImage(cmd_buf->handle(), m_bilateral_blur.image[0]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
            vkCmdClearColorImage(cmd_buf->handle(), m_disocclusion_blur.image->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        }

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur.pipeline->handle());

        BilateralBlurPushConstants push_constants;

        push_constants.z_buffer_params = m_common_resources->z_buffer_params;
        push_constants.direction       = glm::ivec2(1, 0);
        push_constants.radius          = m_bilateral_blur.blur_radius;
        push_constants.g_buffer_mip    = m_g_buffer_mip;

        vkCmdPushConstants(cmd_buf->handle(), m_bilateral_blur.layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_bilateral_blur.write_ds[0]->handle(),
            m_disocclusion_blur.write_ds->handle(),
            m_temporal_accumulation.output_read_ds[m_common_resources->ping_pong]->handle(),
            m_temporal_accumulation.read_ds[m_common_resources->ping_pong]->handle(),
            m_g_buffer->output_ds()->handle(),
            m_temporal_accumulation.indirect_buffer_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur.layout->handle(), 0, 6, descriptor_sets, 0, nullptr);

        vkCmdDispatchIndirect(cmd_buf->handle(), m_temporal_accumulation.denoise_dispatch_args_buffer->handle(), 0);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_bilateral_blur.image[0]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // Horizontal
    {
        DW_SCOPED_SAMPLE("Horizontal", cmd_buf);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_bilateral_blur.image[1]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        {
            VkClearColorValue color;

            color.float32[0] = 1.0f;
            color.float32[1] = 1.0f;
            color.float32[2] = 1.0f;
            color.float32[3] = 1.0f;

            vkCmdClearColorImage(cmd_buf->handle(), m_bilateral_blur.image[1]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        }

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur.pipeline->handle());

        BilateralBlurPushConstants push_constants;

        push_constants.z_buffer_params = m_common_resources->z_buffer_params;
        push_constants.direction       = glm::ivec2(0, 1);
        push_constants.radius          = m_bilateral_blur.blur_radius;
        push_constants.g_buffer_mip    = m_g_buffer_mip;

        vkCmdPushConstants(cmd_buf->handle(), m_bilateral_blur.layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_bilateral_blur.write_ds[1]->handle(),
            m_disocclusion_blur.write_ds->handle(),
            m_bilateral_blur.read_ds[0]->handle(),
            m_temporal_accumulation.read_ds[m_common_resources->ping_pong]->handle(),
            m_g_buffer->output_ds()->handle(),
            m_temporal_accumulation.indirect_buffer_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur.layout->handle(), 0, 6, descriptor_sets, 0, nullptr);

        vkCmdDispatchIndirect(cmd_buf->handle(), m_temporal_accumulation.denoise_dispatch_args_buffer->handle(), 0);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_bilateral_blur.image[1]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------