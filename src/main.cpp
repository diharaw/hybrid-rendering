#include "hybrid_rendering.h"

TemporalReprojection::TemporalReprojection(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height)
{
    m_scale = float(m_sample->m_gpu_resources->g_buffer_1->width()) / float(m_input_width);

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_write_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_read_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_write_ds[i]       = m_sample->m_vk_backend->allocate_descriptor_set(m_write_ds_layout);
        m_read_ds[i]        = m_sample->m_vk_backend->allocate_descriptor_set(m_read_ds_layout);
        m_output_read_ds[i] = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);
    }

    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_write_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_read_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants));

        m_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/temporal_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_color_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_color_image[i]->set_name(m_name + " Reprojection Color");

        m_color_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_color_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_color_view[i]->set_name(m_name + " Reprojection Color");

        m_history_length_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_history_length_image[i]->set_name(m_name + " Reprojection History");

        m_history_length_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_history_length_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_history_length_view[i]->set_name(m_name + " Reprojection History");
    }

    // Reprojection write
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
                storage_image_info.imageView   = m_color_view[i]->handle();
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

            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_history_length_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection read
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

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_color_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_history_length_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection output read
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

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_color_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_output_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

TemporalReprojection::~TemporalReprojection()
{
}

void TemporalReprojection::reproject(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input, dw::vk::DescriptorSet::Ptr prev_input)
{
    clear_images(cmd_buf);

    DW_SCOPED_SAMPLE(m_name + " Temporal Reprojection", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_color_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());

    PushConstants push_constants;

    push_constants.alpha                 = m_alpha;
    push_constants.neighborhood_scale    = m_neighborhood_scale;
    push_constants.use_variance_clipping = (uint32_t)m_use_variance_clipping;
    push_constants.use_tonemap = (uint32_t)m_use_tone_map;

    vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_write_ds[m_sample->m_ping_pong]->handle(),
        m_scale == 1.0f ? m_sample->m_gpu_resources->g_buffer_ds[m_sample->m_ping_pong]->handle() : m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_scale == 1.0f ? m_sample->m_gpu_resources->g_buffer_ds[!m_sample->m_ping_pong]->handle() : m_sample->m_gpu_resources->downsampled_g_buffer_ds[!m_sample->m_ping_pong]->handle(),
        input->handle(),
        (prev_input == nullptr) ? m_output_read_ds[!m_sample->m_ping_pong]->handle() : prev_input->handle(),
        m_read_ds[!m_sample->m_ping_pong]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout->handle(), 0, 6, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_color_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void TemporalReprojection::gui()
{
    ImGui::Checkbox("Variance Clipping", &m_use_variance_clipping);
    ImGui::Checkbox("Tonemap", &m_use_tone_map);
    ImGui::SliderFloat("Neighborhood Scale", &m_neighborhood_scale, 0.0f, 30.0f);
    ImGui::InputFloat("Alpha", &m_alpha);
}

void TemporalReprojection::clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    if (m_sample->m_first_frame)
    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkClearColorValue color;

        color.float32[0] = 0.0f;
        color.float32[1] = 0.0f;
        color.float32[2] = 0.0f;
        color.float32[3] = 0.0f;

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_color_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdClearColorImage(cmd_buf->handle(), m_history_length_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_color_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_color_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }
}

dw::vk::DescriptorSet::Ptr TemporalReprojection::output_ds()
{
    return m_output_read_ds[m_sample->m_ping_pong];
}

SpatialReconstruction::SpatialReconstruction(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height)
{
    m_scale = float(m_sample->m_gpu_resources->g_buffer_1->width()) / float(m_input_width);

    m_image = dw::vk::Image::create(sample->m_vk_backend, VK_IMAGE_TYPE_2D, input_width * 2, input_height * 2, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_image->set_name(name + " Reconstructed");

    m_image_view = dw::vk::ImageView::create(sample->m_vk_backend, m_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_image_view->set_name(name + " Reconstructed");

    m_write_ds = sample->m_vk_backend->allocate_descriptor_set(sample->m_gpu_resources->storage_image_ds_layout);
    m_read_ds  = sample->m_vk_backend->allocate_descriptor_set(sample->m_gpu_resources->combined_sampler_ds_layout);

    // Reconstructed write
    {
        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_image_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &storage_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_write_ds->handle();

        vkUpdateDescriptorSets(sample->m_vk_backend->device(), 1, &write_data, 0, nullptr);
    }

    // Reconstructed read
    {
        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = sample->m_vk_backend->bilinear_sampler()->handle();
        sampler_image_info.imageView   = m_image_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &sampler_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_read_ds->handle();

        vkUpdateDescriptorSets(sample->m_vk_backend->device(), 1, &write_data, 0, nullptr);
    }

    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(sample->m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->per_frame_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants));

        m_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/spatial_reconstruction.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_layout);
        comp_desc.set_shader_stage(module, "main");

        m_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);
    }
}

SpatialReconstruction::~SpatialReconstruction()
{
}

void SpatialReconstruction::reconstruct(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    DW_SCOPED_SAMPLE(m_name + " Reconstruction", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_image->handle(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        subresource_range);

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());

    PushConstants push_constants;

    float z_buffer_params_x = -1.0 + (m_sample->m_near_plane / m_sample->m_far_plane);

    push_constants.z_buffer_params = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / m_sample->m_near_plane, 1.0f / m_sample->m_near_plane);
    push_constants.num_frames      = m_sample->m_num_frames;

    vkCmdPushConstants(cmd_buf->handle(), m_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_sample->m_ubo_size * m_sample->m_vk_backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_write_ds->handle(),
        input->handle(),
        m_scale == 2.0f ? m_sample->m_gpu_resources->g_buffer_ds[m_sample->m_ping_pong]->handle() : m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->per_frame_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_layout->handle(), 0, 4, descriptor_sets, 1, &dynamic_offset);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_image->width()) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_image->height()) / float(NUM_THREADS))), 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

void SpatialReconstruction::gui()
{
}

dw::vk::DescriptorSet::Ptr SpatialReconstruction::output_ds()
{
    return m_read_ds;
}

BilateralBlur::BilateralBlur(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height)
{
    m_scale = float(m_sample->m_gpu_resources->g_buffer_1->width()) / float(m_input_width);

    m_image = dw::vk::Image::create(sample->m_vk_backend, VK_IMAGE_TYPE_2D, input_width, input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_image->set_name(name + " Bilateral");

    m_image_view = dw::vk::ImageView::create(sample->m_vk_backend, m_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_image_view->set_name(name + " Bilateral");

    m_write_ds = sample->m_vk_backend->allocate_descriptor_set(sample->m_gpu_resources->storage_image_ds_layout);
    m_read_ds  = sample->m_vk_backend->allocate_descriptor_set(sample->m_gpu_resources->combined_sampler_ds_layout);

    // write
    {
        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_image_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &storage_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_write_ds->handle();

        vkUpdateDescriptorSets(sample->m_vk_backend->device(), 1, &write_data, 0, nullptr);
    }

    // read
    {
        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = sample->m_vk_backend->bilinear_sampler()->handle();
        sampler_image_info.imageView   = m_image_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &sampler_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_read_ds->handle();

        vkUpdateDescriptorSets(sample->m_vk_backend->device(), 1, &write_data, 0, nullptr);
    }

    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(sample->m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants));

        m_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/bilateral_blur.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_layout);
        comp_desc.set_shader_stage(module, "main");

        m_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);
    }
}

BilateralBlur::~BilateralBlur()
{
}

void BilateralBlur::blur(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    DW_SCOPED_SAMPLE(m_name + " Bilateral Blur", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_image->handle(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        subresource_range);

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());

    PushConstants push_constants;

    float z_buffer_params_x = -1.0 + (m_sample->m_near_plane / m_sample->m_far_plane);

    push_constants.z_buffer_params     = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / m_sample->m_near_plane, 1.0f / m_sample->m_near_plane);
    push_constants.variance_threshold  = m_variance_threshold;
    push_constants.roughness_sigma_min = m_roughness_sigma_min;
    push_constants.roughness_sigma_max = m_roughness_sigma_max;
    push_constants.radius              = m_blur_radius;
    push_constants.roughness_weight    = (uint32_t)m_use_roughness_weight;
    push_constants.depth_weight        = (uint32_t)m_use_depth_weight;
    push_constants.normal_weight       = (uint32_t)m_use_normal_weight;

    vkCmdPushConstants(cmd_buf->handle(), m_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_write_ds->handle(),
        input->handle(),
        m_scale == 1.0f ? m_sample->m_gpu_resources->g_buffer_ds[m_sample->m_ping_pong]->handle() : m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_image->width()) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_image->height()) / float(NUM_THREADS))), 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

void BilateralBlur::prepare_first_frame(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_image->handle(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

void BilateralBlur::gui()
{
    ImGui::Checkbox("Depth Weight", &m_use_depth_weight);
    ImGui::Checkbox("Normal Weight", &m_use_normal_weight);
    ImGui::Checkbox("Roughness Weight", &m_use_roughness_weight);
    ImGui::SliderInt("Radius", &m_blur_radius, 1, 10);
    ImGui::SliderFloat("Variance Threshold", &m_variance_threshold, 0.0f, 1.0f);
    ImGui::InputFloat("Roughness Sigma Min", &m_roughness_sigma_min);
    ImGui::InputFloat("Roughness Sigma Max", &m_roughness_sigma_max);
}

dw::vk::DescriptorSet::Ptr BilateralBlur::output_ds()
{
    return m_read_ds;
}

SVGFDenoiser::SVGFDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height, uint32_t filter_iterations) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height), m_a_trous_filter_iterations(filter_iterations)
{
    create_reprojection_resources();
    create_filter_moments_resources();
    create_a_trous_filter_resources();
}

SVGFDenoiser::~SVGFDenoiser()
{
}

dw::vk::DescriptorSet::Ptr SVGFDenoiser::output_ds()
{
    return m_a_trous_read_ds[m_read_idx];
}

void SVGFDenoiser::denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    clear_images(cmd_buf);
    reprojection(cmd_buf, input);
    filter_moments(cmd_buf);
    a_trous_filter(cmd_buf);
}

void SVGFDenoiser::create_reprojection_resources()
{
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_reprojection_write_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_reprojection_read_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_reprojection_write_ds[i] = m_sample->m_vk_backend->allocate_descriptor_set(m_reprojection_write_ds_layout);
        m_reprojection_read_ds[i]  = m_sample->m_vk_backend->allocate_descriptor_set(m_reprojection_read_ds_layout);
    }

    m_prev_reprojection_read_ds = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);

    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_reprojection_write_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_reprojection_read_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReprojectionPushConstants));

        m_reprojection_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/svgf_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_reprojection_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_reprojection_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_reprojection_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_reprojection_image[i]->set_name(m_name + " Reprojection Color");

        m_reprojection_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_reprojection_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_reprojection_view[i]->set_name(m_name + " Reprojection Color");

        m_moments_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_moments_image[i]->set_name(m_name + " Reprojection Moments");

        m_moments_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_moments_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_moments_view[i]->set_name(m_name + " Reprojection Moments");

        m_history_length_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_history_length_image[i]->set_name(m_name + " Reprojection History");

        m_history_length_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_history_length_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_history_length_view[i]->set_name(m_name + " Reprojection History");
    }

    m_prev_reprojection_image = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_prev_reprojection_image->set_name(m_name + " Previous Reprojection");

    m_prev_reprojection_view = dw::vk::ImageView::create(m_sample->m_vk_backend, m_prev_reprojection_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_prev_reprojection_view->set_name(m_name + " Previous Reprojection");

    // Reprojection write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(6);
        write_datas.reserve(6);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_reprojection_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_reprojection_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_moments_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_reprojection_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_history_length_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 2;
                write_data.dstSet          = m_reprojection_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(6);
        write_datas.reserve(6);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_reprojection_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_reprojection_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_moments_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_reprojection_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_history_length_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 2;
                write_data.dstSet          = m_reprojection_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Previous Reprojection read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_prev_reprojection_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_prev_reprojection_read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void SVGFDenoiser::create_filter_moments_resources()
{
    m_filter_moments_write_ds = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->storage_image_ds_layout);
    m_filter_moments_read_ds  = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);

    dw::vk::PipelineLayout::Desc desc;

    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->storage_image_ds_layout);
    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
    desc.add_descriptor_set_layout(m_reprojection_read_ds_layout);

    desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FilterMomentsPushConstants));

    m_filter_moments_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

    dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/svgf_filter_moments.comp.spv");

    dw::vk::ComputePipeline::Desc comp_desc;

    comp_desc.set_pipeline_layout(m_filter_moments_pipeline_layout);
    comp_desc.set_shader_stage(module, "main");

    m_filter_moments_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);

    m_filter_moments_image = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_filter_moments_image->set_name(m_name + " Filter Moments");

    m_filter_moments_view = dw::vk::ImageView::create(m_sample->m_vk_backend, m_filter_moments_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_filter_moments_view->set_name(m_name + " Filter Moments");

    // Filter Moments write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_filter_moments_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        image_infos.push_back(storage_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_filter_moments_write_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Filter Moments read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_filter_moments_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_filter_moments_read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void SVGFDenoiser::create_a_trous_filter_resources()
{
    for (int i = 0; i < 2; i++)
    {
        m_a_trous_read_ds[i]  = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        m_a_trous_write_ds[i] = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->storage_image_ds_layout);
    }

    dw::vk::PipelineLayout::Desc desc;

    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->storage_image_ds_layout);
    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
    desc.add_descriptor_set_layout(m_reprojection_read_ds_layout);

    desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ATrousFilterPushConstants));

    m_a_trous_filter_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

    dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/svgf_a_trous_filter.comp.spv");

    dw::vk::ComputePipeline::Desc comp_desc;

    comp_desc.set_pipeline_layout(m_a_trous_filter_pipeline_layout);
    comp_desc.set_shader_stage(module, "main");

    m_a_trous_filter_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);

    for (int i = 0; i < 2; i++)
    {
        m_a_trous_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_a_trous_image[i]->set_name(m_name + " A-Trous Filter");

        m_a_trous_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_a_trous_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_a_trous_view[i]->set_name(m_name + " A-Trous Filter");
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
            storage_image_info.imageView   = m_a_trous_view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_a_trous_write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
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

            sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_a_trous_view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_a_trous_read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void SVGFDenoiser::clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    if (m_sample->m_first_frame)
    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkClearColorValue color;

        color.float32[0] = 0.0f;
        color.float32[1] = 0.0f;
        color.float32[2] = 0.0f;
        color.float32[3] = 0.0f;

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_moments_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdClearColorImage(cmd_buf->handle(), m_moments_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_history_length_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_reprojection_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_prev_reprojection_image->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_moments_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }
}

void SVGFDenoiser::reprojection(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    DW_SCOPED_SAMPLE(m_name + " SVGF Temporal Reprojection", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_reprojection_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_moments_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reprojection_pipeline->handle());

    ReprojectionPushConstants push_constants;

    push_constants.alpha         = m_alpha;
    push_constants.moments_alpha = m_moments_alpha;

    vkCmdPushConstants(cmd_buf->handle(), m_reprojection_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_reprojection_write_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[!m_sample->m_ping_pong]->handle(),
        input->handle(),
        m_prev_reprojection_read_ds->handle(),
        m_reprojection_read_ds[!m_sample->m_ping_pong]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reprojection_pipeline_layout->handle(), 0, 6, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

    if (!m_use_spatial_for_feedback)
    {
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresource_range);

        VkImageCopy image_copy_region {};
        image_copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy_region.srcSubresource.layerCount = 1;
        image_copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy_region.dstSubresource.layerCount = 1;
        image_copy_region.extent.width              = m_input_width;
        image_copy_region.extent.height             = m_input_height;
        image_copy_region.extent.depth              = 1;

        // Issue the copy command
        vkCmdCopyImage(
            cmd_buf->handle(),
            m_reprojection_image[m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &image_copy_region);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_reprojection_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_moments_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void SVGFDenoiser::filter_moments(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE(m_name + " SVGF Filter Moments", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_filter_moments_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_filter_moments_pipeline->handle());

    FilterMomentsPushConstants push_constants;

    push_constants.phi_color  = m_phi_color;
    push_constants.phi_normal = m_phi_normal;

    vkCmdPushConstants(cmd_buf->handle(), m_filter_moments_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_filter_moments_write_ds->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_reprojection_read_ds[m_sample->m_ping_pong]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_filter_moments_pipeline_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_filter_moments_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void SVGFDenoiser::a_trous_filter(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE(m_name + " SVGF A-Trous Filter", cmd_buf);

    const uint32_t NUM_THREADS = 32;

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous_filter_pipeline->handle());

    bool    ping_pong = false;
    int32_t read_idx  = 0;
    int32_t write_idx = 1;

    for (int i = 0; i < m_a_trous_filter_iterations; i++)
    {
        read_idx  = (int32_t)ping_pong;
        write_idx = (int32_t)!ping_pong;

        if (i == 0)
        {
            std::vector<VkMemoryBarrier> memory_barriers = {
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
            };

            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous_image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        else
        {
            std::vector<VkMemoryBarrier> memory_barriers = {
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
            };

            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous_image[read_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
                image_memory_barrier(m_a_trous_image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        ATrousFilterPushConstants push_constants;

        push_constants.radius     = m_a_trous_radius;
        push_constants.step_size  = 1 << i;
        push_constants.phi_color  = m_phi_color;
        push_constants.phi_normal = m_phi_normal;

        vkCmdPushConstants(cmd_buf->handle(), m_a_trous_filter_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_a_trous_write_ds[write_idx]->handle(),
            i == 0 ? m_filter_moments_read_ds->handle() : m_a_trous_read_ds[read_idx]->handle(),
            m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
            m_reprojection_read_ds[!m_sample->m_ping_pong]->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous_filter_pipeline_layout->handle(), 0, 4, descriptor_sets, 0, nullptr);

        vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

        ping_pong = !ping_pong;

        if (m_use_spatial_for_feedback && m_a_trous_feedback_iteration == i)
        {
            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_a_trous_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_prev_reprojection_image->handle(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresource_range);

            VkImageCopy image_copy_region {};
            image_copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_copy_region.srcSubresource.layerCount = 1;
            image_copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_copy_region.dstSubresource.layerCount = 1;
            image_copy_region.extent.width              = m_input_width;
            image_copy_region.extent.height             = m_input_height;
            image_copy_region.extent.depth              = 1;

            // Issue the copy command
            vkCmdCopyImage(
                cmd_buf->handle(),
                m_a_trous_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_prev_reprojection_image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &image_copy_region);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_a_trous_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_prev_reprojection_image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }
    }

    m_read_idx = write_idx;

    std::vector<VkMemoryBarrier> memory_barriers = {
        memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    std::vector<VkImageMemoryBarrier> image_barriers = {
        image_memory_barrier(m_a_trous_image[write_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

DiffuseDenoiser::DiffuseDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height)
{
    m_temporal_reprojection = std::unique_ptr<TemporalReprojection>(new TemporalReprojection(sample, name, input_width, input_height));
    m_bilateral_blur         = std::unique_ptr<BilateralBlur>(new BilateralBlur(sample, name, input_width, input_height));

    m_temporal_reprojection->set_variance_clipping(false);
    m_temporal_reprojection->set_neighborhood_scale(3.5f);
    m_temporal_reprojection->set_alpha(0.01f);
    m_bilateral_blur->set_blur_radius(5);
}

DiffuseDenoiser::~DiffuseDenoiser()
{
}

void DiffuseDenoiser::denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    if (m_sample->m_first_frame)
        m_bilateral_blur->prepare_first_frame(cmd_buf);

    m_temporal_reprojection->reproject(cmd_buf, input, m_use_blur_as_temporal_input ? m_bilateral_blur->output_ds() : nullptr);
    m_bilateral_blur->blur(cmd_buf, m_temporal_reprojection->output_ds());
}

void DiffuseDenoiser::gui()
{
    ImGui::Checkbox("Use Blur as Temporal Input", &m_use_blur_as_temporal_input);
    m_temporal_reprojection->gui();
    m_bilateral_blur->gui();
}

dw::vk::DescriptorSet::Ptr DiffuseDenoiser::output_ds()
{
    return m_bilateral_blur->output_ds();
}

ReflectionDenoiser::ReflectionDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height)
{
    m_spatial_reconstruction = std::unique_ptr<SpatialReconstruction>(new SpatialReconstruction(sample, name, input_width, input_height));
    m_temporal_pre_pass      = std::unique_ptr<TemporalReprojection>(new TemporalReprojection(sample, name, input_width * 2, input_height * 2));
    m_temporal_main_pass     = std::unique_ptr<TemporalReprojection>(new TemporalReprojection(sample, name, input_width * 2, input_height * 2));
    m_bilateral_blur         = std::unique_ptr<BilateralBlur>(new BilateralBlur(sample, name, input_width * 2, input_height * 2));

    m_use_blur_as_temporal_input = true;
    m_temporal_pre_pass->set_variance_clipping(true);
    m_temporal_pre_pass->set_neighborhood_scale(3.5f);
    m_temporal_pre_pass->set_alpha(0.05f);
    m_temporal_main_pass->set_variance_clipping(true);
    m_bilateral_blur->set_blur_radius(1);
}

ReflectionDenoiser::~ReflectionDenoiser()
{
}

void ReflectionDenoiser::denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    if (m_sample->m_first_frame)
        m_bilateral_blur->prepare_first_frame(cmd_buf);

    m_spatial_reconstruction->reconstruct(cmd_buf, input);
    
    if (m_use_temporal_pre_pass)
        m_temporal_pre_pass->reproject(cmd_buf, m_spatial_reconstruction->output_ds(), (m_use_blur_as_temporal_input && m_use_bilateral_blur) ? m_bilateral_blur->output_ds() : nullptr);
        
    m_temporal_main_pass->reproject(cmd_buf, m_use_temporal_pre_pass ? m_temporal_pre_pass->output_ds() : m_spatial_reconstruction->output_ds(), m_use_blur_as_temporal_input ? m_bilateral_blur->output_ds() : nullptr);

    if (m_use_bilateral_blur)
        m_bilateral_blur->blur(cmd_buf, m_temporal_main_pass->output_ds());
}

void ReflectionDenoiser::gui()
{
    ImGui::Checkbox("Use Blur as Temporal Input", &m_use_blur_as_temporal_input);
    {
        //ImGui::Text("Spatial Reconstruction");
        //m_spatial_reconstruction->gui();
    }
    {
        ImGui::PushID("TemporalPrePass");
        ImGui::Separator();
        ImGui::Checkbox("Enable", &m_use_temporal_pre_pass);
        ImGui::Text("Temporal Pre Pass");
        m_temporal_pre_pass->gui();
        ImGui::PopID();
    }
    {
        ImGui::PushID("TemporalMainPass");
        ImGui::Separator();
        ImGui::Text("Temporal Main Pass");
        m_temporal_main_pass->gui();
        ImGui::PopID();
    }
    {
        ImGui::Separator();
        ImGui::Text("Bilateral Blur");
        ImGui::Checkbox("Enable", &m_use_bilateral_blur);
        m_bilateral_blur->gui();
    }
}

dw::vk::DescriptorSet::Ptr ReflectionDenoiser::output_ds()
{
    if (m_use_bilateral_blur)
        return m_bilateral_blur->output_ds();
    else
        return m_temporal_main_pass->output_ds();
}

DW_DECLARE_MAIN(HybridRendering)