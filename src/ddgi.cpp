#include "ddgi.h"
#include "utilities.h"
#include "g_buffer.h"
#include <stdexcept>
#include <logger.h>
#include <profiler.h>
#include <imgui.h>
#include <macros.h>
#include <gtc/quaternion.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

// -----------------------------------------------------------------------------------------------------------------------------------

struct DDGIUniforms
{
    glm::vec3  grid_start_position;
    glm::vec3  grid_step;
    glm::ivec3 probe_counts;
    float      max_distance;
    float      depth_sharpness;
    float      hysteresis;
    float      normal_bias;
    float      energy_preservation;
    int        irradiance_probe_side_length;
    int        irradiance_texture_width;
    int        irradiance_texture_height;
    int        depth_probe_side_length;
    int        depth_texture_width;
    int        depth_texture_height;
    int        rays_per_probe;
    int        visibility_test;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct RayTracePushConstants
{
    glm::mat4 random_orientation;
    uint32_t  num_frames;
    uint32_t  infinite_bounces;
    float     gi_intensity;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct ProbeUpdatePushConstants
{
    uint32_t first_frame;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct SampleProbeGridPushConstants
{
    int   g_buffer_mip;
    float gi_intensity;
};

// -----------------------------------------------------------------------------------------------------------------------------------

DDGI::DDGI(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, RayTraceScale scale) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer), m_scale(scale)
{
    auto vk_backend = m_backend.lock();

    float scale_divisor = powf(2.0f, float(scale));

    m_width  = vk_backend->swap_chain_extents().width / scale_divisor;
    m_height = vk_backend->swap_chain_extents().height / scale_divisor;

    m_g_buffer_mip = static_cast<uint32_t>(scale);

    m_random_generator       = std::mt19937(m_random_device());
    m_random_distribution_zo = std::uniform_real_distribution<float>(0.0f, 1.0f);
    m_random_distribution_no = std::uniform_real_distribution<float>(-1.0f, 1.0f);

    create_descriptor_sets();
    create_pipelines();
}

// -----------------------------------------------------------------------------------------------------------------------------------

DDGI::~DDGI()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("DDGI", cmd_buf);

    // If the scene has changed re-initialize the probe grid
    if (m_last_scene_id != m_common_resources->current_scene()->id())
        initialize_probe_grid();

    update_properties_ubo();
    ray_trace(cmd_buf);
    probe_update(cmd_buf);
    sample_probe_grid(cmd_buf);

    m_first_frame = false;
    m_ping_pong   = !m_ping_pong;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::gui()
{
    ImGui::Text("Grid Size: [%i, %i, %i]", m_probe_grid.probe_counts.x, m_probe_grid.probe_counts.y, m_probe_grid.probe_counts.z);
    ImGui::Text("Probe Count: %i", m_probe_grid.probe_counts.x * m_probe_grid.probe_counts.y * m_probe_grid.probe_counts.z);
    ImGui::Checkbox("Visibility Test", &m_probe_grid.visibility_test);
    ImGui::Checkbox("Infinite Bounces", &m_ray_trace.infinite_bounces);

    if (ImGui::InputInt("Rays Per Probe", &m_ray_trace.rays_per_probe))
        recreate_probe_grid_resources();
    if (ImGui::InputFloat("Probe Distance", &m_probe_grid.probe_distance))
        initialize_probe_grid();
    ImGui::InputFloat("Hysteresis", &m_probe_update.hysteresis);
    ImGui::SliderFloat("Infinite Bounce Intensity", &m_ray_trace.infinite_bounce_intensity, 0.0f, 10.0f);
    ImGui::SliderFloat("GI Intensity", &m_sample_probe_grid.gi_intensity, 0.0f, 10.0f);
    ImGui::SliderFloat("Normal Bias", &m_probe_update.normal_bias, 0.0f, 10.0f);
    ImGui::InputFloat("Depth Sharpness", &m_probe_update.depth_sharpness);
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::DescriptorSet::Ptr DDGI::output_ds()
{
    return m_sample_probe_grid.read_ds;
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::DescriptorSet::Ptr DDGI::current_read_ds()
{
    return m_probe_grid.read_ds[static_cast<uint32_t>(!m_ping_pong)];
}

// -----------------------------------------------------------------------------------------------------------------------------------

uint32_t DDGI::current_ubo_offset()
{
    auto vk_backend = m_backend.lock();
    return m_probe_grid.properties_ubo_size * vk_backend->current_frame_idx();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::initialize_probe_grid()
{
    // Get the min and max extents of the scene.
    glm::vec3 min_extents = m_common_resources->current_scene()->min_extents();
    glm::vec3 max_extents = m_common_resources->current_scene()->max_extents();

    // Compute the scene length.
    glm::vec3 scene_length = max_extents - min_extents;

    // Compute the number of probes along each axis.
    // Add 2 more probes to fully cover scene.
    m_probe_grid.probe_counts        = glm::ivec3(scene_length / m_probe_grid.probe_distance) + glm::ivec3(2);
    m_probe_grid.grid_start_position = min_extents;
    m_probe_update.max_distance      = m_probe_grid.probe_distance * 1.5f;

    // Assign current scene ID
    m_last_scene_id = m_common_resources->current_scene()->id();

    recreate_probe_grid_resources();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::create_images()
{
    auto backend = m_backend.lock();

    uint32_t total_probes = m_probe_grid.probe_counts.x * m_probe_grid.probe_counts.y * m_probe_grid.probe_counts.z;

    // Ray Trace
    {
        m_ray_trace.radiance_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_ray_trace.rays_per_probe, total_probes, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_ray_trace.radiance_image->set_name("DDGI Ray Trace Radiance");

        m_ray_trace.radiance_view = dw::vk::ImageView::create(backend, m_ray_trace.radiance_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_ray_trace.radiance_view->set_name("DDGI Ray Trace Radiance");

        m_ray_trace.direction_depth_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_ray_trace.rays_per_probe, total_probes, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_ray_trace.direction_depth_image->set_name("DDGI Ray Trace Direction Depth");

        m_ray_trace.direction_depth_view = dw::vk::ImageView::create(backend, m_ray_trace.direction_depth_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_ray_trace.direction_depth_view->set_name("DDGI Ray Trace Direction Depth");
    }

    // Probe Grid
    {
        // 1-pixel of padding surrounding each probe, 1-pixel padding surrounding entire texture for alignment.
        const int irradiance_width  = (m_probe_grid.irradiance_oct_size + 2) * m_probe_grid.probe_counts.x * m_probe_grid.probe_counts.y + 2;
        const int irradiance_height = (m_probe_grid.irradiance_oct_size + 2) * m_probe_grid.probe_counts.z + 2;

        const int depth_width  = (m_probe_grid.depth_oct_size + 2) * m_probe_grid.probe_counts.x * m_probe_grid.probe_counts.y + 2;
        const int depth_height = (m_probe_grid.depth_oct_size + 2) * m_probe_grid.probe_counts.z + 2;

        for (int i = 0; i < 2; i++)
        {
            m_probe_grid.irradiance_image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, irradiance_width, irradiance_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
            m_probe_grid.irradiance_image[i]->set_name("DDGI Irradiance Probe Grid " + std::to_string(i));

            m_probe_grid.irradiance_view[i] = dw::vk::ImageView::create(backend, m_probe_grid.irradiance_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            m_probe_grid.irradiance_view[i]->set_name("DDGI Irradiance Probe Grid " + std::to_string(i));

            m_probe_grid.depth_image[i] = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, depth_width, depth_height, 1, 1, 1, VK_FORMAT_R16G16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
            m_probe_grid.depth_image[i]->set_name("DDGI Depth Probe Grid " + std::to_string(i));

            m_probe_grid.depth_view[i] = dw::vk::ImageView::create(backend, m_probe_grid.depth_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            m_probe_grid.depth_view[i]->set_name("DDGI Depth Probe Grid " + std::to_string(i));
        }
    }

    // Sample Probe Grid
    {
        m_sample_probe_grid.image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_sample_probe_grid.image->set_name("DDGI Sample Probe Grid");

        m_sample_probe_grid.image_view = dw::vk::ImageView::create(backend, m_sample_probe_grid.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_sample_probe_grid.image_view->set_name("DDGI Sample Probe Grid");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::create_buffers()
{
    auto backend = m_backend.lock();

    m_probe_grid.properties_ubo_size = backend->aligned_dynamic_ubo_size(sizeof(DDGIUniforms));
    m_probe_grid.properties_ubo      = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, m_probe_grid.properties_ubo_size * dw::vk::Backend::kMaxFramesInFlight, VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::create_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Ray Trace
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        m_ray_trace.write_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        m_ray_trace.read_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
    }

    {
        m_ray_trace.write_ds = backend->allocate_descriptor_set(m_ray_trace.write_ds_layout);
        m_ray_trace.read_ds  = backend->allocate_descriptor_set(m_ray_trace.read_ds_layout);
    }

    // Probe Grid
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        m_probe_grid.write_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_probe_grid.write_ds[i] = backend->allocate_descriptor_set(m_probe_grid.write_ds_layout);
        m_probe_grid.read_ds[i]  = backend->allocate_descriptor_set(m_common_resources->ddgi_read_ds_layout);
    }

    // Sample Probe Grid
    {
        m_sample_probe_grid.write_ds = backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        m_sample_probe_grid.write_ds->set_name("DDGI Sample Probe Grid");

        m_sample_probe_grid.read_ds = backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
        m_sample_probe_grid.read_ds->set_name("DDGI Sample Probe Grid");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::write_descriptor_sets()
{
    auto backend = m_backend.lock();

    // Ray Trace Write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_ray_trace.radiance_view->handle();
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
        }

        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_ray_trace.direction_depth_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_ray_trace.write_ds->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Ray Trace Read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_ray_trace.radiance_view->handle();
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
        }

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_ray_trace.direction_depth_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_ray_trace.read_ds->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Probe Grid Write
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
            storage_image_info.imageView   = m_probe_grid.irradiance_view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_probe_grid.write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_probe_grid.depth_view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_probe_grid.write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Probe Grid Read
    for (int i = 0; i < 2; i++)
    {
        std::vector<VkDescriptorBufferInfo> buffer_infos;
        std::vector<VkDescriptorImageInfo>  image_infos;
        std::vector<VkWriteDescriptorSet>   write_datas;
        VkWriteDescriptorSet                write_data;

        image_infos.reserve(2);
        write_datas.reserve(3);

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->bilinear_sampler()->handle();
            sampler_image_info.imageView   = m_probe_grid.irradiance_view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_probe_grid.read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = backend->bilinear_sampler()->handle();
            sampler_image_info.imageView   = m_probe_grid.depth_view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 1;
            write_data.dstSet          = m_probe_grid.read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        {
            VkDescriptorBufferInfo buffer_info;

            buffer_info.range  = sizeof(DDGIUniforms);
            buffer_info.offset = 0;
            buffer_info.buffer = m_probe_grid.properties_ubo->handle();

            buffer_infos.push_back(buffer_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            write_data.pBufferInfo     = &buffer_infos.back();
            write_data.dstBinding      = 2;
            write_data.dstSet          = m_probe_grid.read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Sample Probe Grid write
    {
        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_sample_probe_grid.image_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &storage_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_sample_probe_grid.write_ds->handle();

        vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
    }

    // Sample Probe Grid read
    {
        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = backend->bilinear_sampler()->handle();
        sampler_image_info.imageView   = m_sample_probe_grid.image_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &sampler_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_sample_probe_grid.read_ds->handle();

        vkUpdateDescriptorSets(backend->device(), 1, &write_data, 0, nullptr);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::create_pipelines()
{
    auto vk_backend = m_backend.lock();

    // Ray Trace
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr rgen  = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/gi_ray_trace.rgen.spv");
        dw::vk::ShaderModule::Ptr rchit = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/gi_ray_trace.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/gi_ray_trace.rmiss.spv");

        dw::vk::ShaderBindingTable::Desc sbt_desc;

        sbt_desc.add_ray_gen_group(rgen, "main");
        sbt_desc.add_hit_group(rchit, "main");
        sbt_desc.add_miss_group(rmiss, "main");

        m_ray_trace.sbt = dw::vk::ShaderBindingTable::create(vk_backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_max_pipeline_ray_recursion_depth(1);
        desc.set_shader_binding_table(m_ray_trace.sbt);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_common_resources->current_scene()->descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_ray_trace.write_ds_layout);
        pl_desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_common_resources->skybox_ds_layout);
        pl_desc.add_descriptor_set_layout(m_common_resources->ddgi_read_ds_layout);
        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(RayTracePushConstants));

        m_ray_trace.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, pl_desc);

        desc.set_pipeline_layout(m_ray_trace.pipeline_layout);

        m_ray_trace.pipeline = dw::vk::RayTracingPipeline::create(vk_backend, desc);
    }

    // Probe Update
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_probe_grid.write_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->ddgi_read_ds_layout);
        desc.add_descriptor_set_layout(m_ray_trace.read_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProbeUpdatePushConstants));

        m_probe_update.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
        m_probe_update.pipeline_layout->set_name("Probe Update Pipeline Layout");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_probe_update.pipeline_layout);

        std::string shaders[] = {
            "shaders/gi_irradiance_probe_update.comp.spv",
            "shaders/gi_depth_probe_update.comp.spv"
        };

        for (int i = 0; i < 2; i++)
        {
            dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(vk_backend, shaders[i]);

            comp_desc.set_shader_stage(module, "main");

            m_probe_update.pipeline[i] = dw::vk::ComputePipeline::create(vk_backend, comp_desc);
        }
    }

    // Probe Border Update
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_probe_grid.write_ds_layout);

        m_border_update.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
        m_border_update.pipeline_layout->set_name("Border Update Pipeline Layout");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_border_update.pipeline_layout);

        std::string shaders[] = {
            "shaders/gi_irradiance_border_update.comp.spv",
            "shaders/gi_depth_border_update.comp.spv"
        };

        for (int i = 0; i < 2; i++)
        {
            dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(vk_backend, shaders[i]);

            comp_desc.set_shader_stage(module, "main");

            m_border_update.pipeline[i] = dw::vk::ComputePipeline::create(vk_backend, comp_desc);
        }
    }

    // Sample Probe Grid Update
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->ddgi_read_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SampleProbeGridPushConstants));

        m_sample_probe_grid.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
        m_sample_probe_grid.pipeline_layout->set_name("Sample Probe Grid Pipeline Layout");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_sample_probe_grid.pipeline_layout);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/gi_sample_probe_grid.comp.spv");

        comp_desc.set_shader_stage(module, "main");

        m_sample_probe_grid.pipeline = dw::vk::ComputePipeline::create(vk_backend, comp_desc);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::recreate_probe_grid_resources()
{
    auto backend = m_backend.lock();

    backend->wait_idle();

    m_first_frame = true;

    create_images();
    create_buffers();
    write_descriptor_sets();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::update_properties_ubo()
{
    auto backend = m_backend.lock();

    DDGIUniforms ubo;

    ubo.grid_start_position          = m_probe_grid.grid_start_position;
    ubo.grid_step                    = glm::vec3(m_probe_grid.probe_distance);
    ubo.probe_counts                 = m_probe_grid.probe_counts;
    ubo.max_distance                 = m_probe_update.max_distance;
    ubo.depth_sharpness              = m_probe_update.depth_sharpness;
    ubo.hysteresis                   = m_probe_update.hysteresis;
    ubo.normal_bias                  = m_probe_update.normal_bias;
    ubo.energy_preservation          = m_probe_grid.recursive_energy_preservation;
    ubo.irradiance_probe_side_length = m_probe_grid.irradiance_oct_size;
    ubo.irradiance_texture_width     = m_probe_grid.irradiance_image[0]->width();
    ubo.irradiance_texture_height    = m_probe_grid.irradiance_image[0]->height();
    ubo.depth_probe_side_length      = m_probe_grid.depth_oct_size;
    ubo.depth_texture_width          = m_probe_grid.depth_image[0]->width();
    ubo.depth_texture_height         = m_probe_grid.depth_image[0]->height();
    ubo.rays_per_probe               = m_ray_trace.rays_per_probe;
    ubo.visibility_test              = (int32_t)m_probe_grid.visibility_test;

    uint8_t* ptr = (uint8_t*)m_probe_grid.properties_ubo->mapped_ptr();
    memcpy(ptr + m_probe_grid.properties_ubo_size * backend->current_frame_idx(), &ubo, sizeof(DDGIUniforms));
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Trace", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    uint32_t read_idx = static_cast<uint32_t>(!m_ping_pong);

    if (m_first_frame)
    {
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_probe_grid.irradiance_image[read_idx]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_probe_grid.depth_image[read_idx]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    {
        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_ray_trace.radiance_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_ray_trace.direction_depth_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    }

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_ray_trace.pipeline->handle());

    RayTracePushConstants push_constants;

    push_constants.random_orientation = glm::mat4_cast(glm::angleAxis(m_random_distribution_zo(m_random_generator) * (float(M_PI) * 2.0f), glm::normalize(glm::vec3(m_random_distribution_no(m_random_generator), m_random_distribution_no(m_random_generator), m_random_distribution_no(m_random_generator)))));
    push_constants.num_frames         = m_common_resources->num_frames;
    push_constants.infinite_bounces   = m_ray_trace.infinite_bounces && !m_first_frame ? 1u : 0u;
    push_constants.gi_intensity       = m_ray_trace.infinite_bounce_intensity;

    vkCmdPushConstants(cmd_buf->handle(), m_ray_trace.pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offsets[] = {
        m_common_resources->ubo_size * backend->current_frame_idx(),
        m_probe_grid.properties_ubo_size * backend->current_frame_idx()
    };

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene()->descriptor_set()->handle(),
        m_ray_trace.write_ds->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_common_resources->current_skybox_ds->handle(),
        m_probe_grid.read_ds[static_cast<uint32_t>(!m_ping_pong)]->handle(),
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_ray_trace.pipeline_layout->handle(), 0, 5, descriptor_sets, 2, dynamic_offsets);

    auto& rt_pipeline_props = backend->ray_tracing_pipeline_properties();

    VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
    VkDeviceSize group_stride = group_size;

    const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address() + m_ray_trace.sbt->miss_group_offset(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address() + m_ray_trace.sbt->hit_group_offset(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR callable_sbt = { 0, 0, 0 };

    uint32_t num_total_probes = m_probe_grid.probe_counts.x * m_probe_grid.probe_counts.y * m_probe_grid.probe_counts.z;

    vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, m_ray_trace.rays_per_probe, num_total_probes, 1);

    {
        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_ray_trace.radiance_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_ray_trace.direction_depth_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::probe_update(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Probe Update", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    uint32_t write_idx = static_cast<uint32_t>(m_ping_pong);

    {
        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_probe_grid.irradiance_image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_probe_grid.depth_image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    probe_update(cmd_buf, true);
    probe_update(cmd_buf, false);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, {}, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    border_update(cmd_buf);

    {
        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_probe_grid.irradiance_image[write_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_probe_grid.depth_image[write_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::probe_update(dw::vk::CommandBuffer::Ptr cmd_buf, bool is_irradiance)
{
    DW_SCOPED_SAMPLE(is_irradiance ? "Irradiance" : "Depth", cmd_buf);

    auto backend = m_backend.lock();

    VkPipeline pipeline = m_probe_update.pipeline[1]->handle();

    if (is_irradiance)
        pipeline = m_probe_update.pipeline[0]->handle();

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    ProbeUpdatePushConstants push_constants;

    push_constants.first_frame = (uint32_t)m_first_frame;

    vkCmdPushConstants(cmd_buf->handle(), m_probe_update.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    uint32_t read_idx  = static_cast<uint32_t>(!m_ping_pong);
    uint32_t write_idx = static_cast<uint32_t>(m_ping_pong);

    VkDescriptorSet descriptor_sets[] = {
        m_probe_grid.write_ds[write_idx]->handle(),
        m_probe_grid.read_ds[read_idx]->handle(),
        m_ray_trace.read_ds->handle()
    };

    const uint32_t dynamic_offsets[] = {
        m_probe_grid.properties_ubo_size * backend->current_frame_idx()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_probe_update.pipeline_layout->handle(), 0, 3, descriptor_sets, 1, dynamic_offsets);

    const uint32_t dispatch_x = static_cast<uint32_t>(m_probe_grid.probe_counts.x * m_probe_grid.probe_counts.y);
    const uint32_t dispatch_y = static_cast<uint32_t>(m_probe_grid.probe_counts.z);

    vkCmdDispatch(cmd_buf->handle(), dispatch_x, dispatch_y, 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::border_update(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Border Update", cmd_buf);

    border_update(cmd_buf, true);
    border_update(cmd_buf, false);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::border_update(dw::vk::CommandBuffer::Ptr cmd_buf, bool is_irradiance)
{
    DW_SCOPED_SAMPLE(is_irradiance ? "Irradiance" : "Depth", cmd_buf);

    auto backend = m_backend.lock();

    VkPipeline pipeline = m_border_update.pipeline[1]->handle();

    if (is_irradiance)
        pipeline = m_border_update.pipeline[0]->handle();

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    uint32_t write_idx = static_cast<uint32_t>(m_ping_pong);

    VkDescriptorSet descriptor_sets[] = {
        m_probe_grid.write_ds[write_idx]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_border_update.pipeline_layout->handle(), 0, 1, descriptor_sets, 0, nullptr);

    const uint32_t dispatch_x = static_cast<uint32_t>(m_probe_grid.probe_counts.x * m_probe_grid.probe_counts.y);
    const uint32_t dispatch_y = static_cast<uint32_t>(m_probe_grid.probe_counts.z);

    vkCmdDispatch(cmd_buf->handle(), dispatch_x, dispatch_y, 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::sample_probe_grid(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Sample Probe Grid", cmd_buf);

    auto backend = m_backend.lock();

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_sample_probe_grid.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_sample_probe_grid.pipeline->handle());

    SampleProbeGridPushConstants push_constants;

    push_constants.g_buffer_mip = m_g_buffer_mip;
    push_constants.gi_intensity = m_sample_probe_grid.gi_intensity;

    vkCmdPushConstants(cmd_buf->handle(), m_sample_probe_grid.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offsets[] = {
        m_probe_grid.properties_ubo_size * backend->current_frame_idx(),
        m_common_resources->ubo_size * backend->current_frame_idx()
    };

    VkDescriptorSet descriptor_sets[] = {
        m_sample_probe_grid.write_ds->handle(),
        m_probe_grid.read_ds[static_cast<uint32_t>(m_ping_pong)]->handle(),
        m_g_buffer->output_ds()->handle(),
        m_common_resources->per_frame_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_sample_probe_grid.pipeline_layout->handle(), 0, 4, descriptor_sets, 2, dynamic_offsets);

    const int NUM_THREADS_X = 32;
    const int NUM_THREADS_Y = 32;

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_sample_probe_grid.image->width()) / float(NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(m_sample_probe_grid.image->height()) / float(NUM_THREADS_Y))), 1);

    {
        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_sample_probe_grid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, {}, image_barriers, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------