#include "g_buffer.h"
#include "common_resources.h"
#include <profiler.h>
#include <macros.h>

#define GBUFFER_MIP_LEVELS 9

struct GBufferPushConstants
{
    glm::mat4 model;
    glm::mat4 prev_model;
    uint32_t  material_index;
};

GBuffer::GBuffer(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, uint32_t input_width, uint32_t input_height) :
    m_backend(backend), m_common_resources(common_resources), m_input_width(input_width), m_input_height(input_height)
{
    create_images();
    create_descriptor_set_layouts();
    create_descriptor_sets();
    write_descriptor_sets();
    create_render_pass();
    create_framebuffer();
    create_pipeline();
}

GBuffer::~GBuffer()
{
}

void GBuffer::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("G-Buffer", cmd_buf);

    if (m_common_resources->first_frame)
    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, m_image_linear_z[!m_common_resources->ping_pong]->mip_levels(), 0, 1 };

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_image_linear_z[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        VkClearColorValue color;

        color.float32[0] = 0.0f;
        color.float32[1] = 0.0f;
        color.float32[2] = 0.0f;
        color.float32[3] = 0.0f;

        vkCmdClearColorImage(cmd_buf->handle(), m_image_linear_z[!m_common_resources->ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_image_linear_z[!m_common_resources->ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    VkClearValue clear_values[5];

    clear_values[0].color.float32[0] = 0.0f;
    clear_values[0].color.float32[1] = 0.0f;
    clear_values[0].color.float32[2] = 0.0f;
    clear_values[0].color.float32[3] = 1.0f;

    clear_values[1].color.float32[0] = 0.0f;
    clear_values[1].color.float32[1] = 0.0f;
    clear_values[1].color.float32[2] = 0.0f;
    clear_values[1].color.float32[3] = 1.0f;

    clear_values[2].color.float32[0] = 0.0f;
    clear_values[2].color.float32[1] = 0.0f;
    clear_values[2].color.float32[2] = 0.0f;
    clear_values[2].color.float32[3] = 1.0f;

    clear_values[3].color.float32[0] = 0.0f;
    clear_values[3].color.float32[1] = 0.0f;
    clear_values[3].color.float32[2] = 0.0f;
    clear_values[3].color.float32[3] = 1.0f;

    clear_values[4].color.float32[0] = 1.0f;
    clear_values[4].color.float32[1] = 1.0f;
    clear_values[4].color.float32[2] = 1.0f;
    clear_values[4].color.float32[3] = 1.0f;

    VkRenderPassBeginInfo info    = {};
    info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass               = m_rp->handle();
    info.framebuffer              = m_fbo[m_common_resources->ping_pong]->handle();
    info.renderArea.extent.width  = m_input_width;
    info.renderArea.extent.height = m_input_height;
    info.clearValueCount          = 5;
    info.pClearValues             = &clear_values[0];

    vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp;

    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = (float)m_input_width;
    vp.height   = (float)m_input_height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

    VkRect2D scissor_rect;

    scissor_rect.extent.width  = m_input_width;
    scissor_rect.extent.height = m_input_height;
    scissor_rect.offset.x      = 0;
    scissor_rect.offset.y      = 0;

    vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->handle());

    auto           vk_backend     = m_backend.lock();
    const uint32_t dynamic_offset = m_common_resources->ubo_size * vk_backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene->descriptor_set()->handle(),
        m_common_resources->per_frame_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout->handle(), 0, 2, descriptor_sets, 1, &dynamic_offset);

    const auto& instances = m_common_resources->current_scene->instances();

    for (uint32_t instance_idx = 0; instance_idx < instances.size(); instance_idx++)
    {
        const auto& instance = instances[instance_idx];

        if (!instance.mesh.expired())
        {
            const auto& mesh      = instance.mesh.lock();
            const auto& submeshes = mesh->sub_meshes();

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd_buf->handle(), 0, 1, &mesh->vertex_buffer()->handle(), &offset);
            vkCmdBindIndexBuffer(cmd_buf->handle(), mesh->index_buffer()->handle(), 0, VK_INDEX_TYPE_UINT32);

            for (uint32_t submesh_idx = 0; submesh_idx < submeshes.size(); submesh_idx++)
            {
                auto& submesh = submeshes[submesh_idx];
                auto& mat     = mesh->material(submesh.mat_idx);

                GBufferPushConstants push_constants;

                push_constants.model          = instance.transform;
                push_constants.prev_model     = instance.transform;
                push_constants.material_index = m_common_resources->current_scene->material_index(mat->id());

                vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GBufferPushConstants), &push_constants);

                // Issue draw call.
                vkCmdDrawIndexed(cmd_buf->handle(), submesh.index_count, 1, submesh.base_index, submesh.base_vertex, 0);
            }
        }
    }

    vkCmdEndRenderPass(cmd_buf->handle());

    downsample_gbuffer(cmd_buf);
}

dw::vk::DescriptorSetLayout::Ptr GBuffer::ds_layout()
{
    return m_ds_layout;
}

dw::vk::DescriptorSet::Ptr GBuffer::output_ds()
{
    return m_ds[static_cast<uint32_t>(m_common_resources->ping_pong)];
}

dw::vk::DescriptorSet::Ptr GBuffer::history_ds()
{
    return m_ds[static_cast<uint32_t>(!m_common_resources->ping_pong)];
}

dw::vk::ImageView::Ptr GBuffer::depth_fbo_image_view()
{
    return m_depth_fbo_view;
}

void GBuffer::generate_mipmaps(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::Image::Ptr img, VkImageLayout src_layout, VkImageLayout dst_layout, VkFilter filter, VkImageAspectFlags aspect_flags)
{
    VkImageSubresourceRange initial_subresource_range;
    DW_ZERO_MEMORY(initial_subresource_range);

    initial_subresource_range.aspectMask     = aspect_flags;
    initial_subresource_range.levelCount     = img->mip_levels() - 1;
    initial_subresource_range.layerCount     = 1;
    initial_subresource_range.baseArrayLayer = 0;
    initial_subresource_range.baseMipLevel   = 1;

    dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                        img->handle(),
                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        initial_subresource_range);

    VkImageSubresourceRange subresource_range;
    DW_ZERO_MEMORY(subresource_range);

    subresource_range.aspectMask = aspect_flags;
    subresource_range.levelCount = 1;
    subresource_range.layerCount = 1;

    int32_t mip_width  = m_input_width;
    int32_t mip_height = m_input_height;

    for (int mip_idx = 1; mip_idx < img->mip_levels(); mip_idx++)
    {
        subresource_range.baseMipLevel   = mip_idx - 1;
        subresource_range.baseArrayLayer = 0;

        VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        if (mip_idx == 1)
            layout = src_layout;

        if (layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                                img->handle(),
                                                layout,
                                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                subresource_range);
        }

        VkImageBlit blit                   = {};
        blit.srcOffsets[0]                 = { 0, 0, 0 };
        blit.srcOffsets[1]                 = { mip_width, mip_height, 1 };
        blit.srcSubresource.aspectMask     = aspect_flags;
        blit.srcSubresource.mipLevel       = mip_idx - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount     = 1;
        blit.dstOffsets[0]                 = { 0, 0, 0 };
        blit.dstOffsets[1]                 = { mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1 };
        blit.dstSubresource.aspectMask     = aspect_flags;
        blit.dstSubresource.mipLevel       = mip_idx;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount     = 1;

        vkCmdBlitImage(cmd_buf->handle(),
                       img->handle(),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       img->handle(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &blit,
                       filter);

        dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                            img->handle(),
                                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            dst_layout,
                                            subresource_range);

        if (mip_width > 1) mip_width /= 2;
        if (mip_height > 1) mip_height /= 2;
    }

    subresource_range.baseMipLevel = img->mip_levels() - 1;

    dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                        img->handle(),
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        dst_layout,
                                        subresource_range);
}

void GBuffer::downsample_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Downsample", cmd_buf);

    generate_mipmaps(cmd_buf, m_image_1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FILTER_NEAREST, VK_IMAGE_ASPECT_COLOR_BIT);
    generate_mipmaps(cmd_buf, m_image_2, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FILTER_NEAREST, VK_IMAGE_ASPECT_COLOR_BIT);
    generate_mipmaps(cmd_buf, m_image_3, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FILTER_NEAREST, VK_IMAGE_ASPECT_COLOR_BIT);
    generate_mipmaps(cmd_buf, m_image_linear_z[static_cast<uint32_t>(m_common_resources->ping_pong)], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FILTER_NEAREST, VK_IMAGE_ASPECT_COLOR_BIT);
    generate_mipmaps(cmd_buf, m_depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FILTER_NEAREST, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void GBuffer::create_images()
{
    auto vk_backend = m_backend.lock();

    m_image_1 = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, GBUFFER_MIP_LEVELS, 1, VK_FORMAT_R8G8B8A8_UNORM, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_image_1->set_name("G-Buffer 1 Image");

    m_image_2 = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, GBUFFER_MIP_LEVELS, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_image_2->set_name("G-Buffer 2 Image");

    m_image_3 = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, GBUFFER_MIP_LEVELS, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_image_3->set_name("G-Buffer 3 Image");

    for (int i = 0; i < 2; i++)
    {
        auto image = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, GBUFFER_MIP_LEVELS, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        image->set_name("G-Buffer Linear-Z Image " + std::to_string(i));

        m_image_linear_z[i] = image;

        {
            auto image_view = dw::vk::ImageView::create(vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, GBUFFER_MIP_LEVELS);
            image_view->set_name("G-Buffer Linear-Z Image View " + std::to_string(i));

            m_image_linear_z_view[i] = image_view;
        }

        {
            auto image_view = dw::vk::ImageView::create(vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            image_view->set_name("G-Buffer Linear-Z FBO Image View " + std::to_string(i));

            m_image_linear_z_fbo_view[i] = image_view;
        }
    }

    m_depth = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, GBUFFER_MIP_LEVELS, 1, vk_backend->swap_chain_depth_format(), VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_depth->set_name("G-Buffer Depth Image");

    m_image_1_view = dw::vk::ImageView::create(vk_backend, m_image_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, GBUFFER_MIP_LEVELS);
    m_image_1_view->set_name("G-Buffer 1 Image View");

    m_image_2_view = dw::vk::ImageView::create(vk_backend, m_image_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, GBUFFER_MIP_LEVELS);
    m_image_2_view->set_name("G-Buffer 2 Image View");

    m_image_3_view = dw::vk::ImageView::create(vk_backend, m_image_3, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, GBUFFER_MIP_LEVELS);
    m_image_3_view->set_name("G-Buffer 3 Image View");

    m_depth_view = dw::vk::ImageView::create(vk_backend, m_depth, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 0, GBUFFER_MIP_LEVELS);
    m_depth_view->set_name("G-Buffer Depth Image View");

    m_image_1_fbo_view = dw::vk::ImageView::create(vk_backend, m_image_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_image_1_fbo_view->set_name("G-Buffer 1 FBO Image View");

    m_image_2_fbo_view = dw::vk::ImageView::create(vk_backend, m_image_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_image_2_fbo_view->set_name("G-Buffer 2 FBO Image View");

    m_image_3_fbo_view = dw::vk::ImageView::create(vk_backend, m_image_3, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_image_3_fbo_view->set_name("G-Buffer 3 FBO Image View");

    m_depth_fbo_view = dw::vk::ImageView::create(vk_backend, m_depth, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);
    m_depth_fbo_view->set_name("G-Buffer Depth FBO Image View");
}

void GBuffer::create_descriptor_set_layouts()
{
    dw::vk::DescriptorSetLayout::Desc desc;

    desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
    desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
    desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
    desc.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
    desc.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

    auto vk_backend = m_backend.lock();
    m_ds_layout     = dw::vk::DescriptorSetLayout::create(vk_backend, desc);
    m_ds_layout->set_name("G-Buffer DS Layout");
}

void GBuffer::create_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    for (int i = 0; i < 2; i++)
        m_ds[i] = vk_backend->allocate_descriptor_set(m_ds_layout);
}

void GBuffer::write_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    for (int i = 0; i < 2; i++)
    {
        VkDescriptorImageInfo image_info[5];

        image_info[0].sampler     = vk_backend->nearest_sampler()->handle();
        image_info[0].imageView   = m_image_1_view->handle();
        image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_info[1].sampler     = vk_backend->nearest_sampler()->handle();
        image_info[1].imageView   = m_image_2_view->handle();
        image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_info[2].sampler     = vk_backend->nearest_sampler()->handle();
        image_info[2].imageView   = m_image_3_view->handle();
        image_info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_info[3].sampler     = vk_backend->nearest_sampler()->handle();
        image_info[3].imageView   = m_depth_view->handle();
        image_info[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_info[4].sampler     = vk_backend->nearest_sampler()->handle();
        image_info[4].imageView   = m_image_linear_z_view[i]->handle();
        image_info[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write_data[5];
        DW_ZERO_MEMORY(write_data[0]);
        DW_ZERO_MEMORY(write_data[1]);
        DW_ZERO_MEMORY(write_data[2]);
        DW_ZERO_MEMORY(write_data[3]);
        DW_ZERO_MEMORY(write_data[4]);

        write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[0].descriptorCount = 1;
        write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[0].pImageInfo      = &image_info[0];
        write_data[0].dstBinding      = 0;
        write_data[0].dstSet          = m_ds[i]->handle();

        write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[1].descriptorCount = 1;
        write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[1].pImageInfo      = &image_info[1];
        write_data[1].dstBinding      = 1;
        write_data[1].dstSet          = m_ds[i]->handle();

        write_data[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[2].descriptorCount = 1;
        write_data[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[2].pImageInfo      = &image_info[2];
        write_data[2].dstBinding      = 2;
        write_data[2].dstSet          = m_ds[i]->handle();

        write_data[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[3].descriptorCount = 1;
        write_data[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[3].pImageInfo      = &image_info[3];
        write_data[3].dstBinding      = 3;
        write_data[3].dstSet          = m_ds[i]->handle();

        write_data[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[4].descriptorCount = 1;
        write_data[4].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[4].pImageInfo      = &image_info[4];
        write_data[4].dstBinding      = 4;
        write_data[4].dstSet          = m_ds[i]->handle();

        vkUpdateDescriptorSets(vk_backend->device(), 5, &write_data[0], 0, nullptr);
    }
}

void GBuffer::create_render_pass()
{
    auto vk_backend = m_backend.lock();

    std::vector<VkAttachmentDescription> attachments(5);

    // GBuffer1 attachment
    attachments[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // GBuffer2 attachment
    attachments[1].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // GBuffer3 attachment
    attachments[2].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
    attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // GBuffer Linear Z attachment
    attachments[3].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
    attachments[3].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[3].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // Depth attachment
    attachments[4].format         = vk_backend->swap_chain_depth_format();
    attachments[4].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[4].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[4].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[4].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[4].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[4].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference gbuffer_references[4];

    gbuffer_references[0].attachment = 0;
    gbuffer_references[0].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    gbuffer_references[1].attachment = 1;
    gbuffer_references[1].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    gbuffer_references[2].attachment = 2;
    gbuffer_references[2].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    gbuffer_references[3].attachment = 3;
    gbuffer_references[3].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_reference;
    depth_reference.attachment = 4;
    depth_reference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::vector<VkSubpassDescription> subpass_description(1);

    subpass_description[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_description[0].colorAttachmentCount    = 4;
    subpass_description[0].pColorAttachments       = gbuffer_references;
    subpass_description[0].pDepthStencilAttachment = &depth_reference;
    subpass_description[0].inputAttachmentCount    = 0;
    subpass_description[0].pInputAttachments       = nullptr;
    subpass_description[0].preserveAttachmentCount = 0;
    subpass_description[0].pPreserveAttachments    = nullptr;
    subpass_description[0].pResolveAttachments     = nullptr;

    // Subpass dependencies for layout transitions
    std::vector<VkSubpassDependency> dependencies(2);

    dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass      = 0;
    dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass      = 0;
    dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    m_rp = dw::vk::RenderPass::create(vk_backend, attachments, subpass_description, dependencies);
}

void GBuffer::create_framebuffer()
{
    auto vk_backend = m_backend.lock();

    for (int i = 0; i < 2; i++)
        m_fbo[i] = dw::vk::Framebuffer::create(vk_backend, m_rp, { m_image_1_fbo_view, m_image_2_fbo_view, m_image_3_fbo_view, m_image_linear_z_fbo_view[i], m_depth_fbo_view }, m_input_width, m_input_height, 1);
}

void GBuffer::create_pipeline()
{
    auto vk_backend = m_backend.lock();

    // ---------------------------------------------------------------------------
    // Create shader modules
    // ---------------------------------------------------------------------------

    dw::vk::ShaderModule::Ptr vs = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/g_buffer.vert.spv");
    dw::vk::ShaderModule::Ptr fs = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/g_buffer.frag.spv");

    dw::vk::GraphicsPipeline::Desc pso_desc;

    pso_desc.add_shader_stage(VK_SHADER_STAGE_VERTEX_BIT, vs, "main")
        .add_shader_stage(VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main");

    // ---------------------------------------------------------------------------
    // Create vertex input state
    // ---------------------------------------------------------------------------

    pso_desc.set_vertex_input_state(m_common_resources->meshes[0]->vertex_input_state_desc());

    // ---------------------------------------------------------------------------
    // Create pipeline input assembly state
    // ---------------------------------------------------------------------------

    dw::vk::InputAssemblyStateDesc input_assembly_state_desc;

    input_assembly_state_desc.set_primitive_restart_enable(false)
        .set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    pso_desc.set_input_assembly_state(input_assembly_state_desc);

    // ---------------------------------------------------------------------------
    // Create viewport state
    // ---------------------------------------------------------------------------

    dw::vk::ViewportStateDesc vp_desc;

    vp_desc.add_viewport(0.0f, 0.0f, m_input_width, m_input_height, 0.0f, 1.0f)
        .add_scissor(0, 0, m_input_width, m_input_height);

    pso_desc.set_viewport_state(vp_desc);

    // ---------------------------------------------------------------------------
    // Create rasterization state
    // ---------------------------------------------------------------------------

    dw::vk::RasterizationStateDesc rs_state;

    rs_state.set_depth_clamp(VK_FALSE)
        .set_rasterizer_discard_enable(VK_FALSE)
        .set_polygon_mode(VK_POLYGON_MODE_FILL)
        .set_line_width(1.0f)
        .set_cull_mode(VK_CULL_MODE_BACK_BIT)
        .set_front_face(VK_FRONT_FACE_CLOCKWISE)
        .set_depth_bias(VK_FALSE);

    pso_desc.set_rasterization_state(rs_state);

    // ---------------------------------------------------------------------------
    // Create multisample state
    // ---------------------------------------------------------------------------

    dw::vk::MultisampleStateDesc ms_state;

    ms_state.set_sample_shading_enable(VK_FALSE)
        .set_rasterization_samples(VK_SAMPLE_COUNT_1_BIT);

    pso_desc.set_multisample_state(ms_state);

    // ---------------------------------------------------------------------------
    // Create depth stencil state
    // ---------------------------------------------------------------------------

    dw::vk::DepthStencilStateDesc ds_state;

    ds_state.set_depth_test_enable(VK_TRUE)
        .set_depth_write_enable(VK_TRUE)
        .set_depth_compare_op(VK_COMPARE_OP_LESS)
        .set_depth_bounds_test_enable(VK_FALSE)
        .set_stencil_test_enable(VK_FALSE);

    pso_desc.set_depth_stencil_state(ds_state);

    // ---------------------------------------------------------------------------
    // Create color blend state
    // ---------------------------------------------------------------------------

    dw::vk::ColorBlendAttachmentStateDesc blend_att_desc;

    blend_att_desc.set_color_write_mask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
        .set_blend_enable(VK_FALSE);

    dw::vk::ColorBlendStateDesc blend_state;

    blend_state.set_logic_op_enable(VK_FALSE)
        .set_logic_op(VK_LOGIC_OP_COPY)
        .set_blend_constants(0.0f, 0.0f, 0.0f, 0.0f)
        .add_attachment(blend_att_desc)
        .add_attachment(blend_att_desc)
        .add_attachment(blend_att_desc)
        .add_attachment(blend_att_desc);

    pso_desc.set_color_blend_state(blend_state);

    // ---------------------------------------------------------------------------
    // Create pipeline layout
    // ---------------------------------------------------------------------------

    dw::vk::PipelineLayout::Desc pl_desc;

    pl_desc.add_descriptor_set_layout(m_common_resources->pillars_scene->descriptor_set_layout())
        .add_descriptor_set_layout(m_common_resources->per_frame_ds_layout)
        .add_push_constant_range(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GBufferPushConstants));

    m_pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, pl_desc);

    pso_desc.set_pipeline_layout(m_pipeline_layout);

    // ---------------------------------------------------------------------------
    // Create dynamic state
    // ---------------------------------------------------------------------------

    pso_desc.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
        .add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

    // ---------------------------------------------------------------------------
    // Create pipeline
    // ---------------------------------------------------------------------------

    pso_desc.set_render_pass(m_rp);

    m_pipeline = dw::vk::GraphicsPipeline::create(vk_backend, pso_desc);
}