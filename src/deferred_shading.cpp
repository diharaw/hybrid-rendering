#include "deferred_shading.h"
#include "ray_traced_ao.h"
#include "ray_traced_shadows.h"
#include "ray_traced_reflections.h"
#include "g_buffer.h"
#include "ddgi.h"
#include "utilities.h"
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
    create_render_pass();
    create_framebuffer();
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

    render_shading(cmd_buf, ao, shadows, reflections, ddgi);
    render_skybox(cmd_buf, ddgi);
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

    m_visualize_probe_grid.sphere_mesh = dw::Mesh::load(vk_backend, "mesh/sphere.obj");

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
        0.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f, // bottom-left
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        1.0f,
        0.0f, // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        1.0f,
        1.0f, // top-right
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f, // bottom-left
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f, // top-left
        // front face
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f, // bottom-left
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        0.0f, // bottom-right
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f, // top-right
        -1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f, // top-left
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f, // bottom-left
        // left face
        -1.0f,
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-right
        -1.0f,
        1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f, // top-left
        -1.0f,
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-left
        -1.0f,
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-left
        -1.0f,
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f, // bottom-right
        -1.0f,
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-right
        // right face
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-left
        1.0f,
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f, // bottom-right
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f, // top-left
        1.0f,
        -1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f, // bottom-left
        // bottom face
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f, // top-right
        1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f,
        1.0f, // top-left
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-left
        1.0f,
        -1.0f,
        1.0f,
        0.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-left
        -1.0f,
        -1.0f,
        1.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f, // bottom-right
        -1.0f,
        -1.0f,
        -1.0f,
        0.0f,
        -1.0f,
        0.0f,
        0.0f,
        1.0f, // top-right
        // top face
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f, // top-left
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-right
        1.0f,
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        1.0f, // top-right
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f, // bottom-right
        -1.0f,
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f, // top-left
        -1.0f,
        1.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f // bottom-left
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

void DeferredShading::create_render_pass()
{
    auto vk_backend = m_backend.lock();

    // Shading
    {
        std::vector<VkAttachmentDescription> attachments(1);

        // Deferred attachment
        attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference deferred_reference;

        deferred_reference.attachment = 0;
        deferred_reference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        std::vector<VkSubpassDescription> subpass_description(1);

        subpass_description[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass_description[0].colorAttachmentCount    = 1;
        subpass_description[0].pColorAttachments       = &deferred_reference;
        subpass_description[0].pDepthStencilAttachment = nullptr;
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

        m_shading.rp = dw::vk::RenderPass::create(vk_backend, attachments, subpass_description, dependencies);
    }

    // Skybox
    {
        std::vector<VkAttachmentDescription> attachments(2);

        // Deferred attachment
        attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Depth attachment
        attachments[1].format         = vk_backend->swap_chain_depth_format();
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference deferred_reference;

        deferred_reference.attachment = 0;
        deferred_reference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_reference;
        depth_reference.attachment = 1;
        depth_reference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        std::vector<VkSubpassDescription> subpass_description(1);

        subpass_description[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass_description[0].colorAttachmentCount    = 1;
        subpass_description[0].pColorAttachments       = &deferred_reference;
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

        m_skybox.rp = dw::vk::RenderPass::create(vk_backend, attachments, subpass_description, dependencies);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::create_framebuffer()
{
    auto vk_backend = m_backend.lock();

    // Shading
    {
        m_shading.fbo.reset();
        m_shading.fbo = dw::vk::Framebuffer::create(vk_backend, m_shading.rp, { m_shading.view }, m_width, m_height, 1);
    }

    // Skybox
    for (uint32_t i = 0; i < 2; i++)
    {
        m_skybox.fbo[i].reset();
        m_skybox.fbo[i] = dw::vk::Framebuffer::create(vk_backend, m_skybox.rp, { m_shading.view, m_g_buffer->depth_fbo_image_view(i) }, m_width, m_height, 1);
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

        m_shading.pipeline_layout = dw::vk::PipelineLayout::create(vk_backend, desc);
        m_shading.pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(vk_backend, "shaders/triangle.vert.spv", "shaders/deferred.frag.spv", m_shading.pipeline_layout, m_shading.rp);
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

        struct SkyboxVertex
        {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec2 texcoord;
        };

        vertex_input_state_desc.add_binding_desc(0, sizeof(SkyboxVertex), VK_VERTEX_INPUT_RATE_VERTEX);

        vertex_input_state_desc.add_attribute_desc(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);
        vertex_input_state_desc.add_attribute_desc(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SkyboxVertex, normal));
        vertex_input_state_desc.add_attribute_desc(2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(SkyboxVertex, texcoord));

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
        // Create pipeline
        // ---------------------------------------------------------------------------

        pso_desc.set_render_pass(m_skybox.rp);

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
        // Create pipeline
        // ---------------------------------------------------------------------------

        pso_desc.set_render_pass(m_skybox.rp);

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

    VkClearValue clear_value;

    clear_value.color.float32[0] = 0.0f;
    clear_value.color.float32[1] = 0.0f;
    clear_value.color.float32[2] = 0.0f;
    clear_value.color.float32[3] = 1.0f;

    VkRenderPassBeginInfo info    = {};
    info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass               = m_shading.rp->handle();
    info.framebuffer              = m_shading.fbo->handle();
    info.renderArea.extent.width  = m_width;
    info.renderArea.extent.height = m_height;
    info.clearValueCount          = 1;
    info.pClearValues             = &clear_value;

    vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

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

    vkCmdEndRenderPass(cmd_buf->handle());
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DeferredShading::render_skybox(dw::vk::CommandBuffer::Ptr cmd_buf, DDGI* ddgi)
{
    DW_SCOPED_SAMPLE("Skybox", cmd_buf);

    auto vk_backend = m_backend.lock();

    VkRenderPassBeginInfo info    = {};
    info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass               = m_skybox.rp->handle();
    info.framebuffer              = m_skybox.fbo[m_common_resources->ping_pong]->handle();
    info.renderArea.extent.width  = m_width;
    info.renderArea.extent.height = m_height;
    info.clearValueCount          = 0;
    info.pClearValues             = nullptr;

    vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

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

    vkCmdEndRenderPass(cmd_buf->handle());
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