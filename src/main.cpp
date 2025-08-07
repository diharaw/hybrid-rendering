#include <application.h>
#include <camera.h>
#include <profiler.h>
#include <assimp/scene.h>
#include <equirectangular_to_cubemap.h>
#include <imgui.h>
#include <ImGuizmo.h>
#include <math.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <gtx/matrix_decompose.hpp>
#include <gtc/quaternion.hpp>
#include "g_buffer.h"
#include "deferred_shading.h"
#include "ray_traced_shadows.h"
#include "ray_traced_ao.h"
#include "ray_traced_reflections.h"
#include "ddgi.h"
#include "ground_truth_path_tracer.h"
#include "tone_map.h"
#include "temporal_aa.h"

class HybridRendering : public dw::Application
{
public:
    friend class GBuffer;

protected:
    bool init(int argc, const char* argv[]) override
    {
        m_common_resources         = std::unique_ptr<CommonResources>(new CommonResources(m_vk_backend));
        m_g_buffer                 = std::unique_ptr<GBuffer>(new GBuffer(m_vk_backend, m_common_resources.get(), m_width, m_height));
        m_ray_traced_shadows       = std::unique_ptr<RayTracedShadows>(new RayTracedShadows(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ray_traced_ao            = std::unique_ptr<RayTracedAO>(new RayTracedAO(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ray_traced_reflections   = std::unique_ptr<RayTracedReflections>(new RayTracedReflections(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ddgi                     = std::unique_ptr<DDGI>(new DDGI(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ground_truth_path_tracer = std::unique_ptr<GroundTruthPathTracer>(new GroundTruthPathTracer(m_vk_backend, m_common_resources.get()));
        m_deferred_shading         = std::unique_ptr<DeferredShading>(new DeferredShading(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_temporal_aa              = std::unique_ptr<TemporalAA>(new TemporalAA(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_tone_map                 = std::unique_ptr<ToneMap>(new ToneMap(m_vk_backend, m_common_resources.get()));

        create_camera();
        set_active_scene();

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
        dw::vk::CommandBuffer::Ptr cmd_buf = m_vk_backend->allocate_graphics_command_buffer();

        VkCommandBufferBeginInfo begin_info;
        DW_ZERO_MEMORY(begin_info);

        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(cmd_buf->handle(), &begin_info);

        {
            DW_SCOPED_SAMPLE("Update", cmd_buf);

            debug_gui();

            // Update camera.
            update_camera();

            // Update light.
            update_light_animation();

            // Update uniforms.
            update_uniforms(cmd_buf);

            m_common_resources->current_scene()->build_tlas(cmd_buf);

            update_ibl(cmd_buf);

            // Render.
            m_g_buffer->render(cmd_buf);
            m_ray_traced_shadows->render(cmd_buf);
            m_ray_traced_ao->render(cmd_buf);
            m_ddgi->render(cmd_buf);
            m_ray_traced_reflections->render(cmd_buf, m_ddgi.get());
            m_deferred_shading->render(cmd_buf,
                                       m_ray_traced_ao.get(),
                                       m_ray_traced_shadows.get(),
                                       m_ray_traced_reflections.get(),
                                       m_ddgi.get());
            m_ground_truth_path_tracer->render(cmd_buf);
            m_temporal_aa->render(cmd_buf,
                                  m_deferred_shading.get(),
                                  m_ray_traced_ao.get(),
                                  m_ray_traced_shadows.get(),
                                  m_ray_traced_reflections.get(),
                                  m_ddgi.get(),
                                  m_ground_truth_path_tracer.get(),
                                  m_delta_seconds);
            m_tone_map->render(cmd_buf,
                               m_temporal_aa.get(),
                               m_deferred_shading.get(),
                               m_ray_traced_ao.get(),
                               m_ray_traced_shadows.get(),
                               m_ray_traced_reflections.get(),
                               m_ddgi.get(),
                               m_ground_truth_path_tracer.get(),
                               [this](dw::vk::CommandBuffer::Ptr cmd_buf) {
                                   render_gui(cmd_buf);
                               });

            VkImageSubresourceRange output_subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            m_vk_backend->use_resource(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, m_vk_backend->swapchain_image(), output_subresource_range);

            m_vk_backend->flush_barriers(cmd_buf);
        }

        vkEndCommandBuffer(cmd_buf->handle());

        submit_and_present({ cmd_buf });

        m_common_resources->num_frames++;

        if (m_common_resources->first_frame)
            m_common_resources->first_frame = false;

        m_common_resources->ping_pong = !m_common_resources->ping_pong;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {
        m_tone_map.reset();
        m_temporal_aa.reset();
        m_deferred_shading.reset();
        m_g_buffer.reset();
        m_ground_truth_path_tracer.reset();
        m_ray_traced_shadows.reset();
        m_ray_traced_ao.reset();
        m_ray_traced_reflections.reset();
        m_ddgi.reset();
        m_common_resources.reset();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_pressed(int code) override
    {
        if (m_camera_type == CAMERA_TYPE_FREE)
        {
            // Handle forward movement.
            if (code == GLFW_KEY_W)
                m_heading_speed = m_camera_speed * CAMERA_SPEED_MULTIPLIER;
            else if (code == GLFW_KEY_S)
                m_heading_speed = -m_camera_speed * CAMERA_SPEED_MULTIPLIER;

            // Handle sideways movement.
            if (code == GLFW_KEY_A)
                m_sideways_speed = -m_camera_speed * CAMERA_SPEED_MULTIPLIER;
            else if (code == GLFW_KEY_D)
                m_sideways_speed = m_camera_speed * CAMERA_SPEED_MULTIPLIER;

            if (code == GLFW_KEY_SPACE)
                m_mouse_look = true;
        }

        if (code == GLFW_KEY_G)
            m_debug_gui = !m_debug_gui;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_released(int code) override
    {
        if (m_camera_type == CAMERA_TYPE_FREE)
        {
            // Handle forward movement.
            if (code == GLFW_KEY_W || code == GLFW_KEY_S)
                m_heading_speed = 0.0f;

            // Handle sideways movement.
            if (code == GLFW_KEY_A || code == GLFW_KEY_D)
                m_sideways_speed = 0.0f;
        }

        if (code == GLFW_KEY_SPACE)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_pressed(int code) override
    {
        if (m_camera_type == CAMERA_TYPE_FREE)
        {
            // Enable mouse look.
            if (code == GLFW_MOUSE_BUTTON_RIGHT)
                m_mouse_look = true;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_released(int code) override
    {
        if (m_camera_type == CAMERA_TYPE_FREE)
        {
            // Disable mouse look.
            if (code == GLFW_MOUSE_BUTTON_RIGHT)
                m_mouse_look = false;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    dw::AppSettings intial_app_settings() override
    {
        // Set custom settings here...
        dw::AppSettings settings;

        settings.width       = 1920;
        settings.height      = 1080;
        settings.title       = "Hybrid Rendering";
        settings.ray_tracing = true;
        settings.vsync       = true;

        return settings;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void window_resized(int width, int height) override
    {
        // Override window resized method to update camera projection.
        m_main_camera->update_projection(60.0f, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE, float(m_width) / float(m_height));

        m_vk_backend->wait_idle();

        m_common_resources->write_descriptor_sets(m_vk_backend);
    }

private:
    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_camera()
    {
        m_main_camera                     = std::make_unique<dw::Camera>(60.0f, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE, float(m_width) / float(m_height), glm::vec3(0.0f, 35.0f, 125.0f), glm::vec3(0.0f, 0.0, -1.0f));
        m_common_resources->prev_position = m_main_camera->m_position;

        float z_buffer_params_x             = -1.0 + (CAMERA_NEAR_PLANE / CAMERA_FAR_PLANE);
        m_common_resources->z_buffer_params = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / CAMERA_NEAR_PLANE, 1.0f / CAMERA_NEAR_PLANE);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void debug_gui()
    {
        ImGuizmo::BeginFrame();

        if (m_debug_gui)
        {
            {
                ImGuizmo::SetOrthographic(false);
                ImGuizmo::SetRect(0, 0, m_width, m_height);
                if (ImGuizmo::Manipulate(&m_main_camera->m_view[0][0], &m_main_camera->m_projection[0][0], m_light_transform_operation, ImGuizmo::WORLD, &m_light_transform[0][0], NULL, NULL))
                    m_ground_truth_path_tracer->restart_accumulation();
            }

            bool             open         = true;
            ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_HorizontalScrollbar;

            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(m_width * 0.3f, m_height));

            if (ImGui::Begin("Hybrid Rendering", &open, window_flags))
            {
                if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::TreeNode("General"))
                    {
                        if (ImGui::BeginCombo("Scene", constants::scene_types[m_common_resources->current_scene_type].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::scene_types.size(); i++)
                            {
                                const bool is_selected = (i == m_common_resources->current_scene_type);

                                if (ImGui::Selectable(constants::scene_types[i].c_str(), is_selected))
                                {
                                    m_common_resources->current_scene_type = (SceneType)i;
                                    set_active_scene();
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        if (ImGui::BeginCombo("Environment", constants::environment_types[m_common_resources->current_environment_type].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::environment_types.size(); i++)
                            {
                                const bool is_selected = (i == m_common_resources->current_environment_type);

                                if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY && m_light_type != LIGHT_TYPE_DIRECTIONAL)
                                    continue;

                                if (ImGui::Selectable(constants::environment_types[i].c_str(), is_selected))
                                {
                                    m_common_resources->current_environment_type = (EnvironmentType)i;
                                    m_common_resources->current_skybox_ds        = m_common_resources->skybox_ds[m_common_resources->current_environment_type];
                                    m_ground_truth_path_tracer->restart_accumulation();
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        if (ImGui::BeginCombo("Visualization", constants::visualization_types[m_common_resources->current_visualization_type].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::visualization_types.size(); i++)
                            {
                                const bool is_selected = (i == m_common_resources->current_visualization_type);

                                if (ImGui::Selectable(constants::visualization_types[i].c_str(), is_selected))
                                    m_common_resources->current_visualization_type = (VisualizationType)i;

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_REFLECTIONS)
                        {
                            RayTracedReflections::OutputType type = m_ray_traced_reflections->current_output();

                            if (ImGui::BeginCombo("Buffers", RayTracedReflections::kOutputTypeNames[type].c_str()))
                            {
                                for (uint32_t i = 0; i < RayTracedReflections::kNumOutputTypes; i++)
                                {
                                    const bool is_selected = (i == type);

                                    if (ImGui::Selectable(RayTracedReflections::kOutputTypeNames[i].c_str(), is_selected))
                                        type = (RayTracedReflections::OutputType)i;

                                    if (is_selected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }

                            m_ray_traced_reflections->set_current_output(type);
                        }
                        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_SHADOWS)
                        {
                            RayTracedShadows::OutputType type = m_ray_traced_shadows->current_output();

                            if (ImGui::BeginCombo("Buffers", RayTracedShadows::kOutputTypeNames[type].c_str()))
                            {
                                for (uint32_t i = 0; i < RayTracedShadows::kNumOutputTypes; i++)
                                {
                                    const bool is_selected = (i == type);

                                    if (ImGui::Selectable(RayTracedShadows::kOutputTypeNames[i].c_str(), is_selected))
                                        type = (RayTracedShadows::OutputType)i;

                                    if (is_selected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }

                            m_ray_traced_shadows->set_current_output(type);
                        }
                        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_AMBIENT_OCCLUSION)
                        {
                            RayTracedAO::OutputType type = m_ray_traced_ao->current_output();

                            if (ImGui::BeginCombo("Buffers", RayTracedAO::kOutputTypeNames[type].c_str()))
                            {
                                for (uint32_t i = 0; i < RayTracedAO::kNumOutputTypes; i++)
                                {
                                    const bool is_selected = (i == type);

                                    if (ImGui::Selectable(RayTracedAO::kOutputTypeNames[i].c_str(), is_selected))
                                        type = (RayTracedAO::OutputType)i;

                                    if (is_selected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }

                            m_ray_traced_ao->set_current_output(type);
                        }
                        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_GROUND_TRUTH)
                            m_ground_truth_path_tracer->gui();

                        ImGui::SliderFloat("Roughness Multiplier", &m_common_resources->roughness_multiplier, 0.0f, 1.0f);

                        m_tone_map->gui();

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Light"))
                    {
                        LightType type = m_light_type;

                        if (ImGui::BeginCombo("Type", constants::light_types[type].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::light_types.size(); i++)
                            {
                                const bool is_selected = (i == type);

                                if (ImGui::Selectable(constants::light_types[i].c_str(), is_selected))
                                    type = (LightType)i;

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        if (m_light_type != type)
                        {
                            m_light_type = type;
                            reset_light();
                        }

                        if (m_light_type == LIGHT_TYPE_DIRECTIONAL)
                            directional_light_gui();
                        else if (m_light_type == LIGHT_TYPE_POINT)
                            point_light_gui();
                        else if (m_light_type == LIGHT_TYPE_SPOT)
                            spot_light_gui();

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Camera"))
                    {
                        CameraType type = m_camera_type;

                        if (ImGui::BeginCombo("Type", constants::camera_types[type].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::camera_types.size(); i++)
                            {
                                const bool is_selected = (i == type);

                                if (ImGui::Selectable(constants::camera_types[i].c_str(), is_selected))
                                    type = (CameraType)i;

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        if (type != m_camera_type)
                        {
                            m_camera_type = type;
                            m_ground_truth_path_tracer->restart_accumulation();
                        }

                        if (m_camera_type == CAMERA_TYPE_FIXED)
                        {
                            uint32_t current_angle = m_current_fixed_camera_angle;

                            if (ImGui::BeginCombo("Current Angle", std::to_string(current_angle).c_str()))
                            {
                                for (uint32_t i = 0; i < constants::fixed_camera_forward_vectors[m_common_resources->current_scene_type].size(); i++)
                                {
                                    const bool is_selected = (i == current_angle);

                                    if (ImGui::Selectable(std::to_string(i).c_str(), is_selected))
                                    {
                                        current_angle = i;
                                        m_ground_truth_path_tracer->restart_accumulation();
                                    }

                                    if (is_selected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }

                            m_current_fixed_camera_angle = current_angle;
                        }
                        else if (m_camera_type == CAMERA_TYPE_ANIMATED)
                        {
                            bool is_playing = m_common_resources->demo_players[m_common_resources->current_scene_type]->is_playing();

                            if (ImGui::Checkbox("Is Playing?", &is_playing))
                            {
                                if (is_playing)
                                    m_common_resources->demo_players[m_common_resources->current_scene_type]->play();
                                else
                                    m_common_resources->demo_players[m_common_resources->current_scene_type]->stop();
                            }
                        }

                        if (m_camera_type != CAMERA_TYPE_ANIMATED)
                        {
                            ImGui::SliderFloat("Speed", &m_camera_speed, 0.1f, 10.0f);

                            if (ImGui::Checkbox("Side to Side motion", &m_side_to_side_motion))
                            {
                                if (m_side_to_side_motion)
                                    m_side_to_side_motion_time = 0.0f;

                                m_side_to_side_start_pos = m_main_camera->m_position;
                            }

                            if (m_side_to_side_motion)
                                ImGui::SliderFloat("Side to Side distance", &m_side_to_side_motion_distance, 0.1f, 20.0f);
                        }

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Ray Traced Shadows"))
                    {
                        ImGui::PushID("Ray Traced Shadows");

                        RayTraceScale scale = m_ray_traced_shadows->scale();

                        if (ImGui::BeginCombo("Scale", constants::ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(constants::ray_trace_scales[i].c_str(), is_selected))
                                {
                                    m_vk_backend->wait_idle();
                                    m_ray_traced_shadows.reset();
                                    m_ray_traced_shadows = std::unique_ptr<RayTracedShadows>(new RayTracedShadows(m_vk_backend, m_common_resources.get(), m_g_buffer.get(), (RayTraceScale)i));
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        bool enabled = m_deferred_shading->use_ray_traced_shadows();
                        if (ImGui::Checkbox("Enabled", &enabled))
                            m_deferred_shading->set_use_ray_traced_shadows(enabled);
                        m_ray_traced_shadows->gui();

                        ImGui::PopID();

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Ray Traced Reflections"))
                    {
                        ImGui::PushID("Ray Traced Reflections");

                        RayTraceScale scale = m_ray_traced_reflections->scale();

                        if (ImGui::BeginCombo("Scale", constants::ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(constants::ray_trace_scales[i].c_str(), is_selected))
                                {
                                    m_vk_backend->wait_idle();
                                    m_ray_traced_reflections.reset();
                                    m_ray_traced_reflections = std::unique_ptr<RayTracedReflections>(new RayTracedReflections(m_vk_backend, m_common_resources.get(), m_g_buffer.get(), (RayTraceScale)i));
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        bool enabled = m_deferred_shading->use_ray_traced_reflections();
                        if (ImGui::Checkbox("Enabled", &enabled))
                            m_deferred_shading->set_use_ray_traced_reflections(enabled);

                        m_ray_traced_reflections->gui();

                        ImGui::PopID();

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Ray Traced Ambient Occlusion"))
                    {
                        ImGui::PushID("Ray Traced Ambient Occlusion");

                        RayTraceScale scale = m_ray_traced_ao->scale();

                        if (ImGui::BeginCombo("Scale", constants::ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(constants::ray_trace_scales[i].c_str(), is_selected))
                                {
                                    m_vk_backend->wait_idle();
                                    m_ray_traced_ao.reset();
                                    m_ray_traced_ao = std::unique_ptr<RayTracedAO>(new RayTracedAO(m_vk_backend, m_common_resources.get(), m_g_buffer.get(), (RayTraceScale)i));
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        bool enabled = m_deferred_shading->use_ray_traced_ao();
                        if (ImGui::Checkbox("Enabled", &enabled))
                            m_deferred_shading->set_use_ray_traced_ao(enabled);

                        m_ray_traced_ao->gui();
                        ImGui::PopID();

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Global Illumination"))
                    {
                        ImGui::PushID("GUI_Global_Illumination");

                        RayTraceScale scale = m_ddgi->scale();

                        if (ImGui::BeginCombo("Scale", constants::ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < constants::ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(constants::ray_trace_scales[i].c_str(), is_selected))
                                {
                                    m_vk_backend->wait_idle();
                                    m_ddgi.reset();
                                    m_ddgi = std::unique_ptr<DDGI>(new DDGI(m_vk_backend, m_common_resources.get(), m_g_buffer.get(), (RayTraceScale)i));
                                    set_active_scene();
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        bool enabled = m_deferred_shading->use_ddgi();
                        if (ImGui::Checkbox("Enabled", &enabled))
                            m_deferred_shading->set_use_ddgi(enabled);

                        bool visualize_probe_grid = m_deferred_shading->visualize_probe_grid();
                        if (ImGui::Checkbox("Visualize Probe Grid", &visualize_probe_grid))
                            m_deferred_shading->set_visualize_probe_grid(visualize_probe_grid);

                        m_ddgi->gui();

                        ImGui::PopID();

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("TAA"))
                    {
                        m_temporal_aa->gui();
                        ImGui::TreePop();
                    }
                }
                if (ImGui::CollapsingHeader("Profiler", ImGuiTreeNodeFlags_DefaultOpen))
                    dw::profiler::ui();

                ImGui::End();
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void directional_light_gui()
    {
        m_light_transform_operation = ImGuizmo::OPERATION::ROTATE;

        ImGui::ColorEdit3("Color", &m_light_color.x);
        ImGui::InputFloat("Intensity", &m_light_intensity);
        ImGui::SliderFloat("Radius", &m_light_radius, 0.0f, 0.1f);

        glm::vec3 position;
        glm::vec3 rotation;
        glm::vec3 scale;

        ImGuizmo::DecomposeMatrixToComponents(&m_light_transform[0][0], &position.x, &rotation.x, &scale.x);

        ImGui::InputFloat3("Rotation", &rotation.x);

        ImGuizmo::RecomposeMatrixFromComponents(&position.x, &rotation.x, &scale.x, &m_light_transform[0][0]);

        glm::vec3 out_skew;
        glm::vec4 out_persp;
        glm::vec3 out_scale;
        glm::quat out_orientation;
        glm::vec3 out_position;

        glm::decompose(m_light_transform, out_scale, out_orientation, out_position, out_skew, out_persp);

        ImGui::Checkbox("Animation", &m_light_animation);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void point_light_gui()
    {
        ImGui::ColorEdit3("Color", &m_light_color.x);
        ImGui::InputFloat("Intensity", &m_light_intensity);
        ImGui::SliderFloat("Radius", &m_light_radius, 0.0f, 10.0f);

        m_light_transform_operation = ImGuizmo::TRANSLATE;

        glm::vec3 position;
        glm::vec3 rotation;
        glm::vec3 scale;

        ImGuizmo::DecomposeMatrixToComponents(&m_light_transform[0][0], &position.x, &rotation.x, &scale.x);

        ImGui::InputFloat3("Position", &position.x);

        ImGuizmo::RecomposeMatrixFromComponents(&position.x, &rotation.x, &scale.x, &m_light_transform[0][0]);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void spot_light_gui()
    {
        ImGui::ColorEdit3("Color", &m_light_color.x);
        ImGui::InputFloat("Intensity", &m_light_intensity);
        ImGui::SliderFloat("Radius", &m_light_radius, 0.0f, 10.0f);
        ImGui::SliderFloat("Inner Cone Angle", &m_light_cone_angle_inner, 1.0f, 100.0f);
        ImGui::SliderFloat("Outer Cone Angle", &m_light_cone_angle_outer, 1.0f, 100.0f);

        if (ImGui::RadioButton("Translate", m_light_transform_operation == ImGuizmo::TRANSLATE))
            m_light_transform_operation = ImGuizmo::TRANSLATE;

        ImGui::SameLine();

        if (ImGui::RadioButton("Rotate", m_light_transform_operation == ImGuizmo::ROTATE))
            m_light_transform_operation = ImGuizmo::ROTATE;

        glm::vec3 position;
        glm::vec3 rotation;
        glm::vec3 scale;

        ImGuizmo::DecomposeMatrixToComponents(&m_light_transform[0][0], &position.x, &rotation.x, &scale.x);

        ImGui::InputFloat3("Position", &position.x);
        ImGui::InputFloat3("Rotation", &rotation.x);

        ImGuizmo::RecomposeMatrixFromComponents(&position.x, &rotation.x, &scale.x, &m_light_transform[0][0]);

        if (m_common_resources->current_scene_type == SCENE_TYPE_GLOBAL_ILLUMINATION_TEST)
            ImGui::Checkbox("Animation", &m_light_animation);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void reset_light()
    {
        m_light_transform = glm::mat4(1.0f);

        if (m_common_resources->current_scene_type == SCENE_TYPE_SHADOWS_TEST)
        {
            if (m_light_type == LIGHT_TYPE_DIRECTIONAL)
            {
                m_light_radius    = 0.1f;
                m_light_intensity = 1.0f;

                m_light_transform = glm::rotate(m_light_transform, glm::radians(50.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), glm::radians(50.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_POINT)
            {
                m_light_radius    = 2.5f;
                m_light_intensity = 500.0f;

                m_light_transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 10.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_SPOT)
            {
                m_light_radius           = 2.5f;
                m_light_intensity        = 500.0f;
                m_light_cone_angle_inner = 40.0f;
                m_light_cone_angle_outer = 50.0f;

                glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.5f, 15.0f));

                m_light_transform = T * R;
            }
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_REFLECTIONS_TEST)
        {
            if (m_light_type == LIGHT_TYPE_DIRECTIONAL)
            {
                m_light_radius    = 0.1f;
                m_light_intensity = 1.0f;

                m_light_transform = glm::rotate(m_light_transform, glm::radians(-35.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), glm::radians(-60.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_POINT)
            {
                m_light_radius    = 2.5f;
                m_light_intensity = 500.0f;

                m_light_transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 10.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_SPOT)
            {
                m_light_radius           = 2.5f;
                m_light_intensity        = 5000.0f;
                m_light_cone_angle_inner = 40.0f;
                m_light_cone_angle_outer = 50.0f;

                glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(75.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 15.0f, 20.0f));

                m_light_transform = T * R;
            }
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_GLOBAL_ILLUMINATION_TEST)
        {
            if (m_light_type == LIGHT_TYPE_DIRECTIONAL)
            {
                m_light_radius    = 0.1f;
                m_light_intensity = 1.0f;

                m_light_transform = glm::rotate(m_light_transform, glm::radians(50.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), glm::radians(50.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_POINT)
            {
                m_light_radius    = 2.5f;
                m_light_intensity = 100.0f;

                m_light_transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 4.0f, 2.0f));
            }
            else if (m_light_type == LIGHT_TYPE_SPOT)
            {
                m_light_radius           = 2.5f;
                m_light_intensity        = 1000.0f;
                m_light_cone_angle_inner = 8.0f;
                m_light_cone_angle_outer = 20.0f;

                glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(70.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(-8.25f, 7.5f, 6.0f));

                m_light_transform = T * R;
            }
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_SPONZA)
        {
            if (m_light_type == LIGHT_TYPE_DIRECTIONAL)
            {
                m_light_radius    = 0.08f;
                m_light_intensity = 10.0f;

                m_light_transform = glm::rotate(m_light_transform, glm::radians(30.0f), glm::vec3(0.0f, 0.0f, 1.0f)) * glm::rotate(glm::mat4(1.0f), glm::radians(-10.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_POINT)
            {
                m_light_radius    = 4.0f;
                m_light_intensity = 50000.0f;

                m_light_transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 130.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_SPOT)
            {
                m_light_radius           = 6.5f;
                m_light_intensity        = 500000.0f;
                m_light_cone_angle_inner = 10.0f;
                m_light_cone_angle_outer = 30.0f;

                glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(50.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(80.0f, 60.0f, 15.0f));

                m_light_transform = T * R;
            }
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_PICA_PICA)
        {
            if (m_light_type == LIGHT_TYPE_DIRECTIONAL)
            {
                m_light_radius    = 0.1f;
                m_light_intensity = 1.0f;

                m_light_transform = glm::rotate(m_light_transform, glm::radians(-45.0f), glm::vec3(0.0f, 0.0f, 1.0f)) * glm::rotate(glm::mat4(1.0f), glm::radians(15.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_POINT)
            {
                m_light_radius    = 2.5f;
                m_light_intensity = 500.0f;

                m_light_transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 15.0f, 0.0f));
            }
            else if (m_light_type == LIGHT_TYPE_SPOT)
            {
                m_light_radius           = 2.5f;
                m_light_intensity        = 500.0f;
                m_light_cone_angle_inner = 40.0f;
                m_light_cone_angle_outer = 50.0f;

                glm::mat4 R = glm::rotate(m_light_transform, glm::radians(-30.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(-10.0f, 6.0f, 20.0f));

                m_light_transform = T * R;
            }
        }

        if (m_common_resources->current_environment_type == ENVIRONMENT_TYPE_PROCEDURAL_SKY && m_light_type != LIGHT_TYPE_DIRECTIONAL)
        {
            m_common_resources->current_environment_type = ENVIRONMENT_TYPE_NONE;
            m_common_resources->current_skybox_ds        = m_common_resources->skybox_ds[m_common_resources->current_environment_type];
        }

        m_ground_truth_path_tracer->restart_accumulation();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_uniforms(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Update Uniforms", cmd_buf);

        glm::mat4 current_jitter = glm::translate(glm::mat4(1.0f), glm::vec3(m_temporal_aa->current_jitter(), 0.0f));

        m_common_resources->view                 = m_main_camera->m_view;
        m_common_resources->projection           = m_temporal_aa->enabled() ? current_jitter * m_main_camera->m_projection : m_main_camera->m_projection;
        m_common_resources->prev_view_projection = m_main_camera->m_prev_view_projection;
        m_common_resources->position             = m_main_camera->m_position;

        m_light_direction = glm::normalize(glm::mat3(m_light_transform) * glm::vec3(0.0f, -1.0f, 0.0f));
        m_light_position  = glm::vec3(m_light_transform[3][0], m_light_transform[3][1], m_light_transform[3][2]);

        m_ubo_data.proj_inverse        = glm::inverse(m_common_resources->projection);
        m_ubo_data.view_inverse        = glm::inverse(m_common_resources->view);
        m_ubo_data.view_proj           = m_common_resources->projection * m_common_resources->view;
        m_ubo_data.view_proj_inverse   = glm::inverse(m_ubo_data.view_proj);
        m_ubo_data.prev_view_proj      = m_common_resources->first_frame ? m_common_resources->prev_view_projection : current_jitter * m_common_resources->prev_view_projection;
        m_ubo_data.cam_pos             = glm::vec4(m_common_resources->position, float(m_deferred_shading->use_ray_traced_ao()));
        m_ubo_data.current_prev_jitter = glm::vec4(m_temporal_aa->current_jitter(), m_temporal_aa->prev_jitter());

        m_ubo_data.light.set_light_radius(m_light_radius);
        m_ubo_data.light.set_light_color(m_light_color);
        m_ubo_data.light.set_light_intensity(m_light_intensity);
        m_ubo_data.light.set_light_type(m_light_type);
        m_ubo_data.light.set_light_direction(-m_light_direction);
        m_ubo_data.light.set_light_position(m_light_position);
        m_ubo_data.light.set_light_cos_theta_inner(glm::cos(glm::radians(m_light_cone_angle_inner)));
        m_ubo_data.light.set_light_cos_theta_outer(glm::cos(glm::radians(m_light_cone_angle_outer)));

        m_main_camera->m_prev_view_projection = m_ubo_data.view_proj;

        uint8_t* ptr = (uint8_t*)m_common_resources->ubo->mapped_ptr();
        memcpy(ptr + m_common_resources->ubo_size * m_vk_backend->current_frame_idx(), &m_ubo_data, sizeof(UBO));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_ibl(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        if (m_common_resources->current_environment_type == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
        {
            m_common_resources->sky_environment->hosek_wilkie_sky_model->update(cmd_buf, -m_light_direction);

            {
                DW_SCOPED_SAMPLE("Generate Skybox Mipmap", cmd_buf);
                m_common_resources->sky_environment->hosek_wilkie_sky_model->image()->generate_mipmaps(cmd_buf);
            }

            m_common_resources->sky_environment->cubemap_sh_projection->update(cmd_buf);
            m_common_resources->sky_environment->cubemap_prefilter->update(cmd_buf);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_light_animation()
    {
        if (m_light_animation)
        {
            if (m_common_resources->current_scene_type == SCENE_TYPE_GLOBAL_ILLUMINATION_TEST && m_light_type == LIGHT_TYPE_SPOT)
            {
                float t = sinf(m_light_animation_time) * 0.5f + 0.5f;

                glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(70.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::mix(glm::vec3(-8.25f, 7.5f, 6.0f), glm::vec3(0.25f, 7.5f, 6.0f), t));

                m_light_transform = T * R;
            }
            else if (m_light_type == LIGHT_TYPE_DIRECTIONAL)
            {
                double time = glfwGetTime() * 0.5f;

                m_light_direction.x = sinf(time);
                m_light_direction.z = cosf(time);
                m_light_direction.y = 1.0f;
                m_light_direction   = glm::normalize(m_light_direction);
            }

            m_light_animation_time += m_delta_seconds;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_camera()
    {
        m_temporal_aa->update();

        if (m_camera_type == CAMERA_TYPE_FREE)
        {
            float forward_delta  = m_heading_speed * m_delta;
            float sideways_delta = m_sideways_speed * m_delta;

            m_main_camera->set_translation_delta(m_main_camera->m_forward, forward_delta);
            m_main_camera->set_translation_delta(m_main_camera->m_right, sideways_delta);

            if (forward_delta != 0.0f || sideways_delta != 0.0f)
                m_ground_truth_path_tracer->restart_accumulation();

            m_camera_x = m_mouse_delta_x * m_camera_sensitivity;
            m_camera_y = m_mouse_delta_y * m_camera_sensitivity;

            if (m_mouse_look)
            {
                // Activate Mouse Look
                m_main_camera->set_rotatation_delta(glm::vec3((float)(m_camera_y),
                                                              (float)(m_camera_x),
                                                              (float)(0.0f)));
                m_ground_truth_path_tracer->restart_accumulation();
            }
            else
            {
                m_main_camera->set_rotatation_delta(glm::vec3((float)(0),
                                                              (float)(0),
                                                              (float)(0)));
            }

            if (m_side_to_side_motion)
            {
                m_main_camera->set_position(m_side_to_side_start_pos + m_main_camera->m_right * sinf(static_cast<float>(m_side_to_side_motion_time)) * m_side_to_side_motion_distance);
                m_side_to_side_motion_time += m_delta * 0.005f;
            }

            m_main_camera->update();
        }
        else if (m_camera_type == CAMERA_TYPE_FIXED)
        {
            if (m_side_to_side_motion)
            {
                m_main_camera->update_from_frame(constants::fixed_camera_position_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle] + m_main_camera->m_right * sinf(static_cast<float>(m_side_to_side_motion_time)) * m_side_to_side_motion_distance, constants::fixed_camera_forward_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle], constants::fixed_camera_right_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle]);
                m_side_to_side_motion_time += m_delta * 0.005f;
            }
            else 
                m_main_camera->update_from_frame(constants::fixed_camera_position_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle], constants::fixed_camera_forward_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle], constants::fixed_camera_right_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle]);
        }
        else
            m_common_resources->demo_players[m_common_resources->current_scene_type]->update(m_delta, m_main_camera.get());

        m_common_resources->frame_time    = m_delta_seconds;
        m_common_resources->camera_delta  = m_main_camera->m_position - m_common_resources->prev_position;
        m_common_resources->prev_position = m_main_camera->m_position;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void set_active_scene()
    {
        m_current_fixed_camera_angle = 0;
        m_light_animation_time       = 0.0f;
        m_camera_type                = CAMERA_TYPE_FREE;

        m_common_resources->demo_players[m_common_resources->current_scene_type]->stop();

        if (m_common_resources->current_scene_type == SCENE_TYPE_SHADOWS_TEST)
        {
            m_ddgi->set_normal_bias(1.0f);
            m_ddgi->set_probe_distance(4.0f);
            m_ddgi->set_infinite_bounce_intensity(1.7f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(0.5f);
            m_camera_speed = 2.0f;
            m_main_camera->set_position(glm::vec3(0.321986f, 7.552417f, 28.927477f));
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_REFLECTIONS_TEST)
        {
            m_ddgi->set_normal_bias(1.0f);
            m_ddgi->set_probe_distance(4.0f);
            m_ddgi->set_infinite_bounce_intensity(1.7f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(0.5f);
            m_camera_speed = 0.1f;
            m_main_camera->set_position(glm::vec3(1.449991f, 8.761821f, 33.413113f));
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_GLOBAL_ILLUMINATION_TEST)
        {
            m_ddgi->set_normal_bias(1.0f);
            m_ddgi->set_probe_distance(4.0f);
            m_ddgi->set_infinite_bounce_intensity(0.8f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(0.5f);
            m_light_type   = LIGHT_TYPE_SPOT;
            m_camera_speed = 0.1f;
            m_main_camera->set_position(glm::vec3(1.628197f, 4.763937f, 4.361343f));
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_SPONZA)
        {
            m_ddgi->set_normal_bias(0.1f);
            m_ddgi->set_probe_distance(50.0f);
            m_ddgi->set_infinite_bounce_intensity(1.7f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(5.0f);
            m_camera_speed = 2.0f;
            m_main_camera->set_position(glm::vec3(279.537201f, 35.164913f, -20.101242f));
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_PICA_PICA)
        {
            m_ddgi->set_normal_bias(1.0f);
            m_ddgi->set_probe_distance(4.0f);
            m_ddgi->set_infinite_bounce_intensity(1.7f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(0.5f);
            m_camera_speed = 1.0f;
            m_main_camera->set_position(glm::vec3(-8.837002f, 8.267305f, 18.703117f));
        }

        reset_light();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    std::unique_ptr<CommonResources>       m_common_resources;
    std::unique_ptr<GBuffer>               m_g_buffer;
    std::unique_ptr<DeferredShading>       m_deferred_shading;
    std::unique_ptr<RayTracedShadows>      m_ray_traced_shadows;
    std::unique_ptr<RayTracedAO>           m_ray_traced_ao;
    std::unique_ptr<RayTracedReflections>  m_ray_traced_reflections;
    std::unique_ptr<DDGI>                  m_ddgi;
    std::unique_ptr<GroundTruthPathTracer> m_ground_truth_path_tracer;
    std::unique_ptr<TemporalAA>            m_temporal_aa;
    std::unique_ptr<ToneMap>               m_tone_map;

    // Camera.
    CameraType                  m_camera_type                = CAMERA_TYPE_FREE;
    uint32_t                    m_current_fixed_camera_angle = 0;
    std::unique_ptr<dw::Camera> m_main_camera;
    bool                        m_mouse_look                   = false;
    float                       m_heading_speed                = 0.0f;
    float                       m_sideways_speed               = 0.0f;
    float                       m_camera_sensitivity           = 0.05f;
    float                       m_camera_speed                 = 2.0f;
    float                       m_offset                       = 0.1f;
    float                       m_side_to_side_motion_time     = 0.0f;
    float                       m_side_to_side_motion_distance = 5.0f;
    glm::vec3                   m_side_to_side_start_pos       = glm::vec3(0.0f);
    bool                        m_side_to_side_motion          = false;
    bool                        m_debug_gui                    = false;

    // Camera orientation.
    float m_camera_x;
    float m_camera_y;

    // Light
    ImGuizmo::OPERATION m_light_transform_operation = ImGuizmo::OPERATION::ROTATE;
    glm::mat4           m_light_transform           = glm::mat4(1.0f);
    float               m_light_radius              = 0.1f;
    glm::vec3           m_light_direction           = glm::normalize(glm::vec3(0.568f, 0.707f, -0.421f));
    glm::vec3           m_light_position            = glm::vec3(5.0f);
    glm::vec3           m_light_color               = glm::vec3(1.0f);
    float               m_light_intensity           = 1.0f;
    float               m_light_cone_angle_inner    = 40.0f;
    float               m_light_cone_angle_outer    = 50.0f;
    float               m_light_animation_time      = 0.0f;
    bool                m_light_animation           = false;
    LightType           m_light_type                = LIGHT_TYPE_DIRECTIONAL;

    // Uniforms.
    UBO m_ubo_data;
};

DW_DECLARE_MAIN(HybridRendering)