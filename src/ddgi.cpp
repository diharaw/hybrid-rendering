#include "ddgi.h"
#include <stdexcept>
#include <logger.h>
#include <profiler.h>
#include <macros.h>
#include <imgui.h>

// -----------------------------------------------------------------------------------------------------------------------------------

struct VisualizeProbeGridPushConstants
{
    glm::vec3  grid_start_position;
    glm::vec3  grid_step;
    glm::ivec3 probe_counts;
    float      scale;
};

// -----------------------------------------------------------------------------------------------------------------------------------

uint32_t upper_power_of_two(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

// -----------------------------------------------------------------------------------------------------------------------------------

DDGI::DDGI(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources) :
    m_backend(backend), m_common_resources(common_resources)
{
    load_sphere_mesh();
    create_pipeline();
}

// -----------------------------------------------------------------------------------------------------------------------------------

DDGI::~DDGI()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::render(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    // If the scene has changed re-initialize the probe grid
    if (m_last_scene_id != m_common_resources->current_scene->id())
        initialize_probe_grid();


}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::render_probes(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE("DDGI Visualize Probe Grid", cmd_buf);

    const auto& submeshes = m_visualize_probe_grid.sphere_mesh->sub_meshes();

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd_buf->handle(), 0, 1, &m_visualize_probe_grid.sphere_mesh->vertex_buffer()->handle(), &offset);
    vkCmdBindIndexBuffer(cmd_buf->handle(), m_visualize_probe_grid.sphere_mesh->index_buffer()->handle(), 0, VK_INDEX_TYPE_UINT32);

    auto& submesh = submeshes[0];

    VisualizeProbeGridPushConstants push_constants;

    push_constants.grid_start_position = m_grid_start_position;
    push_constants.grid_step           = glm::vec3(m_probe_distance);
    push_constants.probe_counts        = m_probe_counts;
    push_constants.scale               = m_scale;

    vkCmdPushConstants(cmd_buf->handle(), m_visualize_probe_grid.pipeline_layout->handle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constants), &push_constants);

    uint32_t probe_count = m_probe_counts.x * m_probe_counts.y * m_probe_counts.z;

    // Issue draw call.
    vkCmdDrawIndexed(cmd_buf->handle(), submesh.index_count, probe_count, submesh.base_index, submesh.base_vertex, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::gui()
{
    ImGui::InputFloat("Probe Distance", &m_probe_distance);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::load_sphere_mesh()
{
    auto vk_backend                    = m_backend.lock();

    m_visualize_probe_grid.sphere_mesh = dw::Mesh::load(vk_backend, "mesh/sphere.obj");

    if (!m_visualize_probe_grid.sphere_mesh)
    {
        DW_LOG_ERROR("Failed to load mesh");
        throw std::runtime_error("Failed to load sphere mesh");
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::initialize_probe_grid()
{
    // Get the min and max extents of the scene.
    glm::vec3 min_extents = m_common_resources->current_scene->min_extents();
    glm::vec3 max_extents = m_common_resources->current_scene->max_extents();

    // Compute the scene length.
    glm::vec3 scene_length = max_extents - min_extents;

    // Compute the number of probes along each axis.
    // Add 2 more probes to fully cover scene.
    glm::ivec3 probe_counts = glm::ivec3(scene_length / m_probe_distance) + glm::ivec3(2);

    // Round up the probe count to the next power of 2 value.
    m_probe_counts = glm::ivec3(upper_power_of_two(probe_counts.x), upper_power_of_two(probe_counts.y), upper_power_of_two(probe_counts.z));

    // Compute the scale of the actual rounded up probe grid compared to the scene length.
    glm::vec3 adjusted_scene_length = glm::vec3(m_probe_counts - glm::ivec3(1)) * m_probe_distance;
    glm::vec3 adjusted_scale        = adjusted_scene_length / scene_length;

    // Use the scale to find the grid start position
    m_grid_start_position = min_extents * adjusted_scale;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DDGI::create_pipeline()
{
    auto vk_backend = m_backend.lock();

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
        .add_attachment(blend_att_desc)
        .add_attachment(blend_att_desc)
        .add_attachment(blend_att_desc);

    pso_desc.set_color_blend_state(blend_state);

    // ---------------------------------------------------------------------------
    // Create pipeline layout
    // ---------------------------------------------------------------------------

    dw::vk::PipelineLayout::Desc pl_desc;

    pl_desc.add_push_constant_range(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VisualizeProbeGridPushConstants));

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

    pso_desc.set_render_pass(m_common_resources->skybox_rp);

    m_visualize_probe_grid.pipeline = dw::vk::GraphicsPipeline::create(vk_backend, pso_desc);
}

// -----------------------------------------------------------------------------------------------------------------------------------