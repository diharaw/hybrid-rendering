#include "ssa_reflections.h"
#include "hybrid_rendering.h"

SSaReflections::SSaReflections(HybridRendering* sample, uint32_t width, uint32_t height) :
    m_sample(sample), m_width(width), m_height(height)
{
    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipeline();
}

SSaReflections::~SSaReflections()
{
}

void SSaReflections::create_images()
{
    m_color_image = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_color_image->set_name("Reflection RT Color Image");

    m_color_view = dw::vk::ImageView::create(m_sample->m_vk_backend, m_color_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_color_view->set_name("Reflection RT Color Image View");
}

void SSaReflections::create_descriptor_sets()
{
    m_write_ds = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->storage_image_ds_layout);
    m_read_ds  = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);
}

void SSaReflections::write_descriptor_sets()
{
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_color_view->handle();
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

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = m_sample->m_vk_backend->bilinear_sampler()->handle();
        sampler_image_info.imageView   = m_color_view->handle();
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

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void SSaReflections::create_pipeline()
{
    // ---------------------------------------------------------------------------
    // Create shader modules
    // ---------------------------------------------------------------------------

    dw::vk::ShaderModule::Ptr rgen             = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/reflection.rgen.spv");
    dw::vk::ShaderModule::Ptr rchit            = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/reflection.rchit.spv");
    dw::vk::ShaderModule::Ptr rmiss            = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/reflection.rmiss.spv");
    dw::vk::ShaderModule::Ptr rchit_visibility = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/shadow.rchit.spv");
    dw::vk::ShaderModule::Ptr rmiss_visibility = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/shadow.rmiss.spv");

    dw::vk::ShaderBindingTable::Desc sbt_desc;

    sbt_desc.add_ray_gen_group(rgen, "main");
    sbt_desc.add_hit_group(rchit, "main");
    sbt_desc.add_hit_group(rchit_visibility, "main");
    sbt_desc.add_miss_group(rmiss, "main");
    sbt_desc.add_miss_group(rmiss_visibility, "main");

    m_sbt = dw::vk::ShaderBindingTable::create(m_sample->m_vk_backend, sbt_desc);

    dw::vk::RayTracingPipeline::Desc desc;

    desc.set_max_pipeline_ray_recursion_depth(1);
    desc.set_shader_binding_table(m_sbt);

    // ---------------------------------------------------------------------------
    // Create pipeline layout
    // ---------------------------------------------------------------------------

    dw::vk::PipelineLayout::Desc pl_desc;

    pl_desc.add_descriptor_set_layout(m_sample->m_gpu_resources->pillars_scene->descriptor_set_layout());
    pl_desc.add_descriptor_set_layout(m_sample->m_gpu_resources->storage_image_ds_layout);
    pl_desc.add_descriptor_set_layout(m_sample->m_gpu_resources->per_frame_ds_layout);
    pl_desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
    pl_desc.add_descriptor_set_layout(m_sample->m_gpu_resources->pbr_ds_layout);
    pl_desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
    pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(ReflectionsPushConstants));

    m_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, pl_desc);

    desc.set_pipeline_layout(m_pipeline_layout);

    m_pipeline = dw::vk::RayTracingPipeline::create(m_sample->m_vk_backend, desc);
}