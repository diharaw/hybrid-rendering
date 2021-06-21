#pragma once

#include "common_resources.h"

class DDGI
{
public:
    DDGI(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources);
    ~DDGI();

    void render(dw::vk::CommandBuffer::Ptr cmd_buf);
    void render_probes(dw::vk::CommandBuffer::Ptr cmd_buf);
    void gui();

private:
    void load_sphere_mesh();
    void initialize_probe_grid();
    void create_pipeline();

private:
    struct VisualizeProbeGrid
    {
        dw::Mesh::Ptr                 sphere_mesh;
        dw::vk::GraphicsPipeline::Ptr pipeline;
        dw::vk::PipelineLayout::Ptr   pipeline_layout;
    };

    uint32_t                       m_last_scene_id = UINT32_MAX;
    std::weak_ptr<dw::vk::Backend> m_backend;
    CommonResources*               m_common_resources;
    float                          m_probe_distance = 1.0f;
    float                          m_scale          = 1.0f;
    glm::vec3                      m_grid_start_position;
    glm::ivec3                     m_probe_counts;
    VisualizeProbeGrid             m_visualize_probe_grid;
};