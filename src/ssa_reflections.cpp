#include "ssa_reflections.h"
#include "common_resources.h"
#include "g_buffer.h"
#include <profiler.h>
#include <macros.h>
#include <imgui.h>

#define MAX_MIP_LEVELS 8
#define NUM_THREADS_X 32
#define NUM_THREADS_Y 32

// -----------------------------------------------------------------------------------------------------------------------------------

struct RayTracePushConstants
{
    uint32_t num_frames;
    float bias;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct ImagePyramidPushConstants
{
    glm::vec4 z_buffer_params;
    int32_t   fine_g_buffer_mip;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct BlurPushConstants
{
    glm::vec4 z_buffer_params;
    float     radius;
    int32_t   g_buffer_mip;
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
    image_pyramid(cmd_buf);
    blur(cmd_buf);
    resolve(cmd_buf);
    upsample(cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::gui()
{
    ImGui::PushID("SSaReflections");
   
    ImGui::InputFloat("Bias", &m_bias);
    ImGui::SliderInt("Blur Radius", &m_blur.radius, 1, 5);
    
    ImGui::PopID();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::ray_trace(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Ray Trace", cmd_buf);

    auto backend = m_backend.lock();

    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, MAX_MIP_LEVELS, 0, 1 };

        // Transition ray tracing output image back to general layout
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_ray_trace.image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);
    }

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_ray_trace.pipeline->handle());

    RayTracePushConstants push_constants;

    push_constants.num_frames = m_common_resources->num_frames;
    push_constants.bias = m_bias;

    vkCmdPushConstants(cmd_buf->handle(), m_ray_trace.pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_common_resources->current_scene->descriptor_set()->handle(),
        m_ray_trace.write_ds[0]->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_g_buffer->output_ds()->handle(),
        m_common_resources->pbr_ds->handle(),
        m_common_resources->skybox_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_ray_trace.pipeline_layout->handle(), 0, 6, descriptor_sets, 1, &dynamic_offset);

    auto& rt_pipeline_props = backend->ray_tracing_pipeline_properties();

    VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
    VkDeviceSize group_stride = group_size;

    const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
    const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address() + m_ray_trace.sbt->miss_group_offset(), group_stride, group_size * 2 };
    const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_ray_trace.pipeline->shader_binding_table_buffer()->device_address() + m_ray_trace.sbt->hit_group_offset(), group_stride, group_size * 2 };
    const VkStridedDeviceAddressRegionKHR callable_sbt = { 0, 0, 0 };

    vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, m_width, m_height, 1);
    
    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_ray_trace.image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::image_pyramid(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Image Pyramid", cmd_buf);

    uint32_t w = m_ray_trace.image->width() / 2;
    uint32_t h = m_ray_trace.image->height() / 2;

    for (int i = 1; i < MAX_MIP_LEVELS; i++)
    {
        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_image_pyramid.pipeline->handle());

        ImagePyramidPushConstants push_constants;

        push_constants.z_buffer_params = m_common_resources->z_buffer_params;
        push_constants.fine_g_buffer_mip = i;

        vkCmdPushConstants(cmd_buf->handle(), m_image_pyramid.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_ray_trace.write_ds[i]->handle(),
            m_image_pyramid.read_ds[i - 1]->handle(),
            m_g_buffer->output_ds()->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_image_pyramid.pipeline_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

        vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(w) / float(NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(h) / float(NUM_THREADS_Y))), 1);

        {
            VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1 };

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_ray_trace.image->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }

        w /= 2;
        h /= 2;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::blur(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("Blur", cmd_buf);

    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, MAX_MIP_LEVELS, 0, 1 };

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_blur.image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);
    }

    uint32_t w = m_blur.image->width();
    uint32_t h = m_blur.image->height();

    for (int i = 0; i < MAX_MIP_LEVELS; i++)
    {
        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_blur.pipeline->handle());

        BlurPushConstants push_constants;

        push_constants.z_buffer_params   = m_common_resources->z_buffer_params;
        push_constants.radius          = m_blur.radius;
        push_constants.g_buffer_mip = i + 1;

        vkCmdPushConstants(cmd_buf->handle(), m_blur.pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_blur.write_ds[i]->handle(),
            m_image_pyramid.read_ds[i]->handle(),
            m_g_buffer->output_ds()->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_blur.pipeline_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

        vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(w) / float(NUM_THREADS_X))), static_cast<uint32_t>(ceil(float(h) / float(NUM_THREADS_Y))), 1);

        {
            VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1 };

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_blur.image->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }

        w /= 2;
        h /= 2;
    }
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

    // Ray Trace
    {
        m_ray_trace.image = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, MAX_MIP_LEVELS, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_ray_trace.image->set_name("SSa Reflection RT Color Image");

        m_ray_trace.single_image_views.resize(MAX_MIP_LEVELS);

        for (int i = 0; i < MAX_MIP_LEVELS; i++)
        {
            m_ray_trace.single_image_views[i] = dw::vk::ImageView::create(vk_backend, m_ray_trace.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, i, 1);
            m_ray_trace.single_image_views[i]->set_name("SSa Reflection RT Color Single Image View " + std::to_string(i));
        }

        m_ray_trace.all_image_view = dw::vk::ImageView::create(vk_backend, m_ray_trace.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, MAX_MIP_LEVELS);
        m_ray_trace.all_image_view->set_name("SSa Reflection RT Color All Image View"); 
    }

    // Blur
    {
        m_blur.image = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, MAX_MIP_LEVELS, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_blur.image->set_name("SSa Reflection Blur Image");

        m_blur.single_image_views.resize(MAX_MIP_LEVELS);

        for (int i = 0; i < MAX_MIP_LEVELS; i++)
        {
            m_blur.single_image_views[i] = dw::vk::ImageView::create(vk_backend, m_blur.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, i, 1);
            m_blur.single_image_views[i]->set_name("SSa Reflection Blur Write Image View");
        }

        m_blur.all_image_view = dw::vk::ImageView::create(vk_backend, m_blur.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, MAX_MIP_LEVELS);
        m_blur.all_image_view->set_name("SSa Reflection Blur Image View");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::create_descriptor_sets()
{
    auto vk_backend = m_backend.lock();
    
    // Ray Trace
    {
        m_ray_trace.write_ds.resize(MAX_MIP_LEVELS);

        for (int i = 0; i < MAX_MIP_LEVELS; i++)
            m_ray_trace.write_ds[i] = vk_backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);
        
        m_ray_trace.read_ds  = vk_backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
    }

    // Image Pyramid
    {
        m_image_pyramid.read_ds.resize(MAX_MIP_LEVELS);

        for (int i = 0; i < MAX_MIP_LEVELS; i++)
            m_image_pyramid.read_ds[i] = vk_backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
    }

    // Blur
    {
        m_blur.write_ds.resize(MAX_MIP_LEVELS);

        for (int i = 0; i < MAX_MIP_LEVELS; i++)
            m_blur.write_ds[i] = vk_backend->allocate_descriptor_set(m_common_resources->storage_image_ds_layout);

        m_blur.read_ds = vk_backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::write_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    // Ray Trace
    {
        for (int i = 0; i < MAX_MIP_LEVELS; i++)
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(1);
            write_datas.reserve(1);

            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_ray_trace.single_image_views[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_ray_trace.write_ds[i]->handle();

            write_datas.push_back(write_data);

            vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }
    }

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = vk_backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_ray_trace.all_image_view->handle();
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

        vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Image Pyramid
    {
        for (int i = 0; i < MAX_MIP_LEVELS; i++)
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(1);
            write_datas.reserve(1);

            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = vk_backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_ray_trace.single_image_views[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_image_pyramid.read_ds[i]->handle();

            write_datas.push_back(write_data);

            vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }
    }

    // Ray Trace
    {
        for (int i = 0; i < MAX_MIP_LEVELS; i++)
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(1);
            write_datas.reserve(1);

            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_blur.single_image_views[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_blur.write_ds[i]->handle();

            write_datas.push_back(write_data);

            vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }
    }

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = vk_backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_blur.all_image_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_blur.read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void SSaReflections::create_pipeline()
{
    auto vk_backend = m_backend.lock();

    // Ray Trace
    {
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

        m_ray_trace.sbt = dw::vk::ShaderBindingTable::create(vk_backend, sbt_desc);

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
        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(RayTracePushConstants));

        m_ray_trace.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, pl_desc);

        desc.set_pipeline_layout(m_ray_trace.pipeline_layout);

        m_ray_trace.pipeline = dw::vk::RayTracingPipeline::create(vk_backend, desc);
    }

    // Image Pyramid
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ImagePyramidPushConstants));

        m_image_pyramid.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
        m_image_pyramid.pipeline_layout->set_name("Color Pyramid Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/ssa_image_pyramid.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_image_pyramid.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_image_pyramid.pipeline = dw::vk::ComputePipeline::create(vk_backend, comp_desc);
    }

    // Blur
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_common_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPushConstants));

        m_blur.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
        m_blur.pipeline_layout->set_name("Blur Pipeline Layout");

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/ssa_blur.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_blur.pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_blur.pipeline = dw::vk::ComputePipeline::create(vk_backend, comp_desc);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------