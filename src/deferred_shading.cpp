#include "deferred_shading.h"
#include "ray_traced_ao.h"
#include "ray_traced_shadows.h"
#include "ray_traced_reflections.h"
#include "g_buffer.h"
#include "ddgi.h"
#include <profiler.h>
#include <macros.h>
#include <imgui.h>
#include <logger.h>

// -----------------------------------------------------------------------------------------------------------------------------------

struct ShadingPushConstants
{
    int shadows     = 1;
    int ao          = 1;
    int reflections = 1;
    int gi          = 1;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct SkyboxPushConstants
{
    glm::mat4 projection;
    glm::mat4 view;
};

// -----------------------------------------------------------------------------------------------------------------------------------

struct VisualizeProbeGridPushConstants
{
    float scale;
};

// -----------------------------------------------------------------------------------------------------------------------------------

DeferredShading::DeferredShading(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer)
{
    load_sphere_mesh();
    create_cube();
    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipeline();
}

// -----------------------------------------------------------------------------------------------------------------------------------

DeferredShading::~DeferredShading()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::render(dw::vk::CommandBuffer::Ptr cmd_buf,
                             RayTracedAO*               ao,
                             RayTracedShadows*          shadows,
                             RayTracedReflections*      reflections,
                             DDGI*                      ddgi)
{
    DW_SCOPED_SAMPLE("Deferred Shading", cmd_buf);

    auto backend = m_backend.lock();

    render_shading(cmd_buf, ao, shadows, reflections, ddgi);
    render_skybox(cmd_buf, ddgi);

    VkImageSubresourceRange color_subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    backend->use_resource(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_shading.image, color_subresource_range);

    backend->flush_barriers(cmd_buf);
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::DescriptorSet::Ptr DeferredShading::output_ds()
{
    return m_shading.read_ds;
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::vk::Image::Ptr DeferredShading::output_image()
{
    return m_shading.image;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::load_sphere_mesh()
{
    auto vk_backend = m_backend.lock();

    m_visualize_probe_grid.sphere_mesh = dw::Mesh::load(vk_backend, "meshes/sphere.obj");

    if (!m_visualize_probe_grid.sphere_mesh)
    {
        DW_LOG_ERROR("Failed to load mesh");
        throw std::runtime_error("Failed to load sphere mesh");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::create_cube()
{
    float cube_vertices[] = {
        // back face
        -1.0f,
        -1.0f,
        -1.0f,
        // bottom-left
        1.0f,
        1.0f,
        -1.0f,
        // top-right
        1.0f,
        -1.0f,
        -1.0f,
        // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        // top-right
        -1.0f,
        -1.0f,
        -1.0f,
        // bottom-left
        -1.0f,
        1.0f,
        -1.0f,
        // top-left
        // front face
        -1.0f,
        -1.0f,
        1.0f,
        // bottom-left
        1.0f,
        -1.0f,
        1.0f,
        // bottom-right
        1.0f,
        1.0f,
        1.0f,
        // top-right
        1.0f,
        1.0f,
        1.0f,
        // top-right
        -1.0f,
        1.0f,
        1.0f,
        // top-left
        -1.0f,
        -1.0f,
        1.0f,
        // bottom-left
        // left face
        -1.0f,
        1.0f,
        1.0f,
        // top-right
        -1.0f,
        1.0f,
        -1.0f,
        // top-left
        -1.0f,
        -1.0f,
        -1.0f,
        // bottom-left
        -1.0f,
        -1.0f,
        -1.0f,
        // bottom-left
        -1.0f,
        -1.0f,
        1.0f,
        // bottom-right
        -1.0f,
        1.0f,
        1.0f,
        // top-right
        // right face
        1.0f,
        1.0f,
        1.0f,
        // top-left
        1.0f,
        -1.0f,
        -1.0f,
        // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        // top-right
        1.0f,
        -1.0f,
        -1.0f,
        // bottom-right
        1.0f,
        1.0f,
        1.0f,
        // top-left
        1.0f,
        -1.0f,
        1.0f,
        // bottom-left
        // bottom face
        -1.0f,
        -1.0f,
        -1.0f,
        // top-right
        1.0f,
        -1.0f,
        -1.0f,
        // top-left
        1.0f,
        -1.0f,
        1.0f,
        // bottom-left
        1.0f,
        -1.0f,
        1.0f,
        // bottom-left
        -1.0f,
        -1.0f,
        1.0f,
        // bottom-right
        -1.0f,
        -1.0f,
        -1.0f,
        // top-right
        // top face
        -1.0f,
        1.0f,
        -1.0f,
        // top-left
        1.0f,
        1.0f,
        1.0f,
        // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        // top-right
        1.0f,
        1.0f,
        1.0f,
        // bottom-right
        -1.0f,
        1.0f,
        -1.0f,
        // top-left
        -1.0f,
        1.0f,
        1.0f // bottom-left
    };

    auto vk_backend = m_backend.lock();

    m_skybox.cube_vbo = dw::vk::Buffer::create(vk_backend, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, sizeof(cube_vertices), VMA_MEMORY_USAGE_GPU_ONLY, 0, cube_vertices);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::create_images()
{
    auto vk_backend = m_backend.lock();

    m_width  = vk_backend->swap_chain_extents().width;
    m_height = vk_backend->swap_chain_extents().height;

    // Shading
    {
        m_shading.image = dw::vk::Image::create(vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_shading.image->set_name("Deferred Image");

        m_shading.view = dw::vk::ImageView::create(vk_backend, m_shading.image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_shading.view->set_name("Deferred Image View");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::create_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    // Shading
    {
        m_shading.read_ds = vk_backend->allocate_descriptor_set(m_common_resources->combined_sampler_ds_layout);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::write_descriptor_sets()
{
    auto vk_backend = m_backend.lock();

    // Shading Read
    {
        VkDescriptorImageInfo image;

        image.sampler     = vk_backend->bilinear_sampler()->handle();
        image.imageView   = m_shading.view->handle();
        image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write_data;
        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_shading.read_ds->handle();

        vkUpdateDescriptorSets(vk_backend->device(), 1, &write_data, 0, nullptr);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::create_pipeline()
{
    auto vk_backend = m_backend.lock();

    // Shading
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_g_buffer->ds_layout());
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout);
        desc.add_descriptor_set_layout(m_common_resources->skybox_ds_layout);
        desc.add_push_constant_range(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShadingPushConstants));

        VkFormat format = m_shading.image->format();

        m_shading.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
        m_shading.pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(vk_backend, "shaders/triangle.vert.spv", "shaders/deferred.frag.spv", m_shading.pipeline_layout, 1, &format);
    }

    // Skybox
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr vs = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/skybox.vert.spv");
        dw::vk::ShaderModule::Ptr fs = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/skybox.frag.spv");

        dw::vk::GraphicsPipeline::Desc pso_desc;

        pso_desc.add_shader_stage(VK_SHADER_STAGE_VERTEX_BIT, vs, "main")
            .add_shader_stage(VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main");

        // ---------------------------------------------------------------------------
        // Create vertex input state
        // ---------------------------------------------------------------------------

        dw::vk::VertexInputStateDesc vertex_input_state_desc;

        vertex_input_state_desc.add_binding_desc(0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX);

        vertex_input_state_desc.add_attribute_desc(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);

        pso_desc.set_vertex_input_state(vertex_input_state_desc);

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

        vp_desc.add_viewport(0.0f, 0.0f, m_width, m_height, 0.0f, 1.0f)
            .add_scissor(0, 0, m_width, m_height);

        pso_desc.set_viewport_state(vp_desc);

        // ---------------------------------------------------------------------------
        // Create rasterization state
        // ---------------------------------------------------------------------------

        dw::vk::RasterizationStateDesc rs_state;

        rs_state.set_depth_clamp(VK_FALSE)
            .set_rasterizer_discard_enable(VK_FALSE)
            .set_polygon_mode(VK_POLYGON_MODE_FILL)
            .set_line_width(1.0f)
            .set_cull_mode(VK_CULL_MODE_NONE)
            .set_front_face(VK_FRONT_FACE_COUNTER_CLOCKWISE)
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
            .set_depth_write_enable(VK_FALSE)
            .set_depth_compare_op(VK_COMPARE_OP_LESS_OR_EQUAL)
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
            .add_attachment(blend_att_desc);

        pso_desc.set_color_blend_state(blend_state);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_common_resources->skybox_ds_layout)
            .add_push_constant_range(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SkyboxPushConstants));

        m_skybox.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, pl_desc);

        pso_desc.set_pipeline_layout(m_skybox.pipeline_layout);

        // ---------------------------------------------------------------------------
        // Create dynamic state
        // ---------------------------------------------------------------------------

        pso_desc.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
            .add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

        // ---------------------------------------------------------------------------
        // Create rendering state
        // ---------------------------------------------------------------------------

        pso_desc.add_color_attachment_format(m_shading.image->format());
        pso_desc.set_depth_attachment_format(vk_backend->swap_chain_depth_format());
        pso_desc.set_stencil_attachment_format(VK_FORMAT_UNDEFINED);

        // ---------------------------------------------------------------------------
        // Create pipeline
        // ---------------------------------------------------------------------------

        m_skybox.pipeline = dw::vk::GraphicsPipeline::create(vk_backend, pso_desc);
    }

    // Visualize Probe Grid

    // Probe Visualization
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr vs = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/gi_probe_visualization.vert.spv");
        dw::vk::ShaderModule::Ptr fs = dw::vk::ShaderModule::create_from_file(vk_backend, "shaders/gi_probe_visualization.frag.spv");

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

        vp_desc.add_viewport(0.0f, 0.0f, vk_backend->swap_chain_extents().width, vk_backend->swap_chain_extents().height, 0.0f, 1.0f)
            .add_scissor(0, 0, vk_backend->swap_chain_extents().width, vk_backend->swap_chain_extents().height);

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
            .add_attachment(blend_att_desc);

        pso_desc.set_color_blend_state(blend_state);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_common_resources->per_frame_ds_layout)
            .add_descriptor_set_layout(m_common_resources->ddgi_read_ds_layout)
            .add_push_constant_range(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(VisualizeProbeGridPushConstants));

        m_visualize_probe_grid.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, pl_desc);

        pso_desc.set_pipeline_layout(m_visualize_probe_grid.pipeline_layout);

        // ---------------------------------------------------------------------------
        // Create dynamic state
        // ---------------------------------------------------------------------------

        pso_desc.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
            .add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

        // ---------------------------------------------------------------------------
        // Create rendering state
        // ---------------------------------------------------------------------------

        pso_desc.add_color_attachment_format(m_shading.image->format());
        pso_desc.set_depth_attachment_format(vk_backend->swap_chain_depth_format());
        pso_desc.set_stencil_attachment_format(VK_FORMAT_UNDEFINED);

        // ---------------------------------------------------------------------------
        // Create pipeline
        // ---------------------------------------------------------------------------

        m_visualize_probe_grid.pipeline = dw::vk::GraphicsPipeline::create(vk_backend, pso_desc);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::render_shading(dw::vk::CommandBuffer::Ptr cmd_buf,
                                     RayTracedAO*               ao,
                                     RayTracedShadows*          shadows,
                                     RayTracedReflections*      reflections,
                                     DDGI*                      ddgi)
{
    DW_SCOPED_SAMPLE("Opaque", cmd_buf);

    auto vk_backend = m_backend.lock();

    VkImageSubresourceRange color_subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    
    vk_backend->use_resource(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, m_shading.image, color_subresource_range);
    
    vk_backend->flush_barriers(cmd_buf);

    VkRenderingAttachmentInfoKHR color_attachment = {};

    color_attachment.sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    color_attachment.imageView        = m_shading.view->handle();
    color_attachment.imageLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp           = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp          = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = { 0.0f, 0.0f, 0.0f, 0.0f };

    VkRenderingInfoKHR rendering_info = {};

    rendering_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    rendering_info.renderArea           = { 0, 0, m_width, m_height };
    rendering_info.layerCount           = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments    = &color_attachment;
    rendering_info.pDepthAttachment     = nullptr;

    vkCmdBeginRenderingKHR(cmd_buf->handle(), &rendering_info);

    VkViewport vp;

    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = (float)m_width;
    vp.height   = (float)m_height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

    VkRect2D scissor_rect;

    scissor_rect.extent.width  = m_width;
    scissor_rect.extent.height = m_height;
    scissor_rect.offset.x      = 0;
    scissor_rect.offset.y      = 0;

    vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_shading.pipeline->handle());

    ShadingPushConstants push_constants;

    push_constants.shadows     = (float)m_shading.use_ray_traced_shadows;
    push_constants.ao          = (float)m_shading.use_ray_traced_ao;
    push_constants.reflections = (float)m_shading.use_ray_traced_reflections;
    push_constants.gi          = (float)m_shading.use_ddgi;

    vkCmdPushConstants(cmd_buf->handle(), m_shading.pipeline_layout->handle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_common_resources->ubo_size * vk_backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_g_buffer->output_ds()->handle(),
        ao->output_ds()->handle(),
        shadows->output_ds()->handle(),
        reflections->output_ds()->handle(),
        ddgi->output_ds()->handle(),
        m_common_resources->per_frame_ds->handle(),
        m_common_resources->current_skybox_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_shading.pipeline_layout->handle(), 0, 7, descriptor_sets, 1, &dynamic_offset);

    vkCmdDraw(cmd_buf->handle(), 3, 1, 0, 0);

    vkCmdEndRendering(cmd_buf->handle());
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::render_skybox(dw::vk::CommandBuffer::Ptr cmd_buf, DDGI* ddgi)
{
    DW_SCOPED_SAMPLE("Skybox", cmd_buf);

    auto vk_backend = m_backend.lock();

    VkImageSubresourceRange color_subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageSubresourceRange depth_subresource_range = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

    vk_backend->use_resource(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, m_shading.image, color_subresource_range);
    vk_backend->use_resource(VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk_backend->swapchain_depth_image(), depth_subresource_range);

    vk_backend->flush_barriers(cmd_buf);

    VkRenderingAttachmentInfoKHR color_attachment = {};

    color_attachment.sType            = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    color_attachment.imageView        = m_shading.view->handle();
    color_attachment.imageLayout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp           = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp          = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = { 0.0f, 0.0f, 0.0f, 0.0f };

    VkRenderingAttachmentInfoKHR depth_stencil_sttachment {};

    depth_stencil_sttachment.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depth_stencil_sttachment.imageView               = vk_backend->swapchain_depth_image_view()->handle();
    depth_stencil_sttachment.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_stencil_sttachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_stencil_sttachment.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
    depth_stencil_sttachment.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfoKHR rendering_info = {};

    rendering_info.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    rendering_info.renderArea           = { 0, 0, m_width, m_height };
    rendering_info.layerCount           = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments    = &color_attachment;
    rendering_info.pDepthAttachment     = &depth_stencil_sttachment;

    vkCmdBeginRenderingKHR(cmd_buf->handle(), &rendering_info);

    VkViewport vp;

    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = (float)m_width;
    vp.height   = (float)m_height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

    VkRect2D scissor_rect;

    scissor_rect.extent.width  = m_width;
    scissor_rect.extent.height = m_height;
    scissor_rect.offset.x      = 0;
    scissor_rect.offset.y      = 0;

    vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

    // Also render the DDGI probe visualization here, if requested.
    render_probes(cmd_buf, ddgi);

    {
        DW_SCOPED_SAMPLE("Skybox", cmd_buf);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_skybox.pipeline->handle());
        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_skybox.pipeline_layout->handle(), 0, 1, &m_common_resources->current_skybox_ds->handle(), 0, nullptr);

        SkyboxPushConstants push_constants;

        push_constants.projection = m_common_resources->projection;
        push_constants.view       = m_common_resources->view;

        vkCmdPushConstants(cmd_buf->handle(), m_skybox.pipeline_layout->handle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constants), &push_constants);

        const VkBuffer     buffer = m_skybox.cube_vbo->handle();
        const VkDeviceSize size   = 0;
        vkCmdBindVertexBuffers(cmd_buf->handle(), 0, 1, &buffer, &size);

        vkCmdDraw(cmd_buf->handle(), 36, 1, 0, 0);
    }

    vkCmdEndRendering(cmd_buf->handle());
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::render_probes(dw::vk::CommandBuffer::Ptr cmd_buf, DDGI* ddgi)
{
    if (m_visualize_probe_grid.enabled)
    {
        DW_SCOPED_SAMPLE("DDGI Visualize Probe Grid", cmd_buf);

        auto vk_backend = m_backend.lock();

        const auto& submeshes = m_visualize_probe_grid.sphere_mesh->sub_meshes();

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_visualize_probe_grid.pipeline->handle());

        const uint32_t dynamic_offsets[] = {
            m_common_resources->ubo_size * vk_backend->current_frame_idx(),
            ddgi->current_ubo_offset()
        };

        VkDescriptorSet descriptor_sets[] = {
            m_common_resources->per_frame_ds->handle(),
            ddgi->current_read_ds()->handle(),
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_visualize_probe_grid.pipeline_layout->handle(), 0, 2, descriptor_sets, 2, dynamic_offsets);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd_buf->handle(), 0, 1, &m_visualize_probe_grid.sphere_mesh->vertex_buffer()->handle(), &offset);
        vkCmdBindIndexBuffer(cmd_buf->handle(), m_visualize_probe_grid.sphere_mesh->index_buffer()->handle(), 0, VK_INDEX_TYPE_UINT32);

        auto& submesh = submeshes[0];

        VisualizeProbeGridPushConstants push_constants;

        push_constants.scale = m_visualize_probe_grid.scale;

        vkCmdPushConstants(cmd_buf->handle(), m_visualize_probe_grid.pipeline_layout->handle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_constants), &push_constants);

        uint32_t probe_count = ddgi->probe_counts().x * ddgi->probe_counts().y * ddgi->probe_counts().z;

        // Issue draw call.
        vkCmdDrawIndexed(cmd_buf->handle(), submesh.index_count, probe_count, submesh.base_index, submesh.base_vertex, 0);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------