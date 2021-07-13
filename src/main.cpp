#include <application.h>
#include <camera.h>
#include <profiler.h>
#include <assimp/scene.h>
#include <equirectangular_to_cubemap.h>
#include "g_buffer.h"
#include "deferred_shading.h"
#include "ray_traced_shadows.h"
#include "ray_traced_ao.h"
#include "ray_traced_reflections.h"
#include "ddgi.h"
#include "temporal_aa.h"
#include "utilities.h"

#define NUM_PILLARS 6
#define CAMERA_NEAR_PLANE 1.0f
#define CAMERA_FAR_PLANE 1000.0f

const std::vector<std::string> environment_map_images = { "textures/Arches_E_PineTree_3k.hdr", "textures/BasketballCourt_3k.hdr", "textures/Etnies_Park_Center_3k.hdr", "textures/LA_Downtown_Helipad_GoldenHour_3k.hdr" };
const std::vector<std::string> environment_types      = { "None", "Procedural Sky", "Arches Pine Tree", "Basketball Court", "Etnies Park Central", "LA Downtown Helipad" };
const std::vector<std::string> visualization_types    = { "Final", "Shadows", "Ambient Occlusion", "Reflections", "Global Illumination" };
const std::vector<std::string> scene_types            = { "Pillars", "Sponza", "Pica Pica" };
const std::vector<std::string> ray_trace_scales       = { "Full-Res", "Half-Res", "Quarter-Res" };

struct Light
{
    glm::vec4  data0;
    glm::vec4  data1;
    glm::ivec4 data2;
};

void set_light_direction(Light& light, glm::vec3 value)
{
    light.data0.x = value.x;
    light.data0.y = value.y;
    light.data0.z = value.z;
}

void set_light_position(Light& light, glm::vec3 value)
{
    light.data0.x = value.x;
    light.data0.y = value.y;
    light.data0.z = value.z;
}

void set_light_color(Light& light, glm::vec3 value)
{
    light.data1.x = value.x;
    light.data1.y = value.y;
    light.data1.z = value.z;
}

void set_light_intensity(Light& light, float value)
{
    light.data0.w = value;
}

void set_light_radius(Light& light, float value)
{
    light.data1.w = value;
}

void set_light_type(Light& light, LightType value)
{
    light.data2.r = value;
}

struct ToneMapPushConstants
{
    int   single_channel;
    float exposure;
};

// Uniform buffer data structure.
struct UBO
{
    DW_ALIGNED(16)
    glm::mat4 view_inverse;
    DW_ALIGNED(16)
    glm::mat4 proj_inverse;
    DW_ALIGNED(16)
    glm::mat4 view_proj_inverse;
    DW_ALIGNED(16)
    glm::mat4 prev_view_proj;
    DW_ALIGNED(16)
    glm::mat4 view_proj;
    DW_ALIGNED(16)
    glm::vec4 cam_pos;
    DW_ALIGNED(16)
    glm::vec4 current_prev_jitter;
    DW_ALIGNED(16)
    Light light;
};

class HybridRendering : public dw::Application
{
public:
    friend class GBuffer;

protected:
    bool init(int argc, const char* argv[]) override
    {
        m_common_resources = std::unique_ptr<CommonResources>(new CommonResources());

        if (!create_uniform_buffer())
            return false;

        // Load mesh.
        if (!load_mesh())
        {
            DW_LOG_INFO("Failed to load mesh");
            return false;
        }

        m_common_resources->brdf_preintegrate_lut = std::unique_ptr<dw::BRDFIntegrateLUT>(new dw::BRDFIntegrateLUT(m_vk_backend));
        m_common_resources->blue_noise_image_1    = dw::vk::Image::create_from_file(m_vk_backend, "texture/LDR_RGBA_0.png");
        m_common_resources->blue_noise_view_1     = dw::vk::ImageView::create(m_vk_backend, m_common_resources->blue_noise_image_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_common_resources->blue_noise_image_2    = dw::vk::Image::create_from_file(m_vk_backend, "texture/LDR_RGBA_1.png");
        m_common_resources->blue_noise_view_2     = dw::vk::ImageView::create(m_vk_backend, m_common_resources->blue_noise_image_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_common_resources->blue_noise            = std::unique_ptr<BlueNoise>(new BlueNoise(m_vk_backend));

        create_environment_resources();
        create_descriptor_set_layouts();
        create_descriptor_sets();
        write_descriptor_sets();

        m_g_buffer               = std::unique_ptr<GBuffer>(new GBuffer(m_vk_backend, m_common_resources.get(), m_width, m_height));
        m_ray_traced_shadows     = std::unique_ptr<RayTracedShadows>(new RayTracedShadows(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ray_traced_ao          = std::unique_ptr<RayTracedAO>(new RayTracedAO(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ddgi                   = std::unique_ptr<DDGI>(new DDGI(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ray_traced_reflections = std::unique_ptr<RayTracedReflections>(new RayTracedReflections(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_deferred_shading       = std::unique_ptr<DeferredShading>(new DeferredShading(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_temporal_aa            = std::unique_ptr<TemporalAA>(new TemporalAA(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));

        create_tone_map_pipeline();
        set_active_scene();

        // Create camera.
        create_camera();

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

            if (m_debug_gui)
            {
                if (ImGui::Begin("Hybrid Rendering"))
                {
                    if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        if (ImGui::BeginCombo("Scene", scene_types[m_common_resources->current_scene_type].c_str()))
                        {
                            for (uint32_t i = 0; i < scene_types.size(); i++)
                            {
                                const bool is_selected = (i == m_common_resources->current_scene_type);

                                if (ImGui::Selectable(scene_types[i].c_str(), is_selected))
                                {
                                    m_common_resources->current_scene_type = (SceneType)i;
                                    set_active_scene();
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        if (ImGui::BeginCombo("Environment", environment_types[m_common_resources->current_environment_type].c_str()))
                        {
                            for (uint32_t i = 0; i < environment_types.size(); i++)
                            {
                                const bool is_selected = (i == m_common_resources->current_environment_type);

                                if (ImGui::Selectable(environment_types[i].c_str(), is_selected))
                                {
                                    m_common_resources->current_environment_type = (EnvironmentType)i;
                                    m_common_resources->current_skybox_ds        = m_common_resources->skybox_ds[m_common_resources->current_environment_type];
                                }

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }

                        if (ImGui::BeginCombo("Visualization", visualization_types[m_common_resources->current_visualization_type].c_str()))
                        {
                            for (uint32_t i = 0; i < visualization_types.size(); i++)
                            {
                                const bool is_selected = (i == m_common_resources->current_visualization_type);

                                if (ImGui::Selectable(visualization_types[i].c_str(), is_selected))
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

                        ImGui::InputFloat("Exposure", &m_exposure);
                    }
                    if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::ColorEdit3("Color", &m_light_color.x);
                        ImGui::InputFloat("Intensity", &m_light_intensity);
                        ImGui::SliderFloat("Radius", &m_light_radius, 0.0f, 0.1f);
                        ImGui::InputFloat3("Direction", &m_light_direction.x);
                        ImGui::Checkbox("Animation", &m_light_animation);
                    }
                    if (ImGui::CollapsingHeader("Ray Traced Shadows", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::PushID("Ray Traced Shadows");

                        RayTraceScale scale = m_ray_traced_shadows->scale();

                        if (ImGui::BeginCombo("Scale", ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(ray_trace_scales[i].c_str(), is_selected))
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
                    }
                    if (ImGui::CollapsingHeader("Ray Traced Reflections", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::PushID("Ray Traced Reflections");

                        RayTraceScale scale = m_ray_traced_reflections->scale();

                        if (ImGui::BeginCombo("Scale", ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(ray_trace_scales[i].c_str(), is_selected))
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
                    }
                    if (ImGui::CollapsingHeader("Ray Traced Ambient Occlusion", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::PushID("Ray Traced Ambient Occlusion");

                        RayTraceScale scale = m_ray_traced_ao->scale();

                        if (ImGui::BeginCombo("Scale", ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(ray_trace_scales[i].c_str(), is_selected))
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
                    }
                    if (ImGui::CollapsingHeader("Global Illumination", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::PushID("GUI_Global_Illumination");

                        RayTraceScale scale = m_ddgi->scale();

                        if (ImGui::BeginCombo("Scale", ray_trace_scales[scale].c_str()))
                        {
                            for (uint32_t i = 0; i < ray_trace_scales.size(); i++)
                            {
                                const bool is_selected = (i == scale);

                                if (ImGui::Selectable(ray_trace_scales[i].c_str(), is_selected))
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
                    }
                    if (ImGui::CollapsingHeader("TAA", ImGuiTreeNodeFlags_DefaultOpen))
                        m_temporal_aa->gui();
                    if (ImGui::CollapsingHeader("Profiler", ImGuiTreeNodeFlags_DefaultOpen))
                        dw::profiler::ui();

                    ImGui::End();
                }
            }

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
            m_temporal_aa->render(cmd_buf,
                                  m_deferred_shading.get(),
                                  m_ray_traced_ao.get(),
                                  m_ray_traced_shadows.get(),
                                  m_ray_traced_reflections.get(),
                                  m_ddgi.get(),
                                  m_delta_seconds);
            tone_map(cmd_buf);
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
        m_temporal_aa.reset();
        m_deferred_shading.reset();
        m_g_buffer.reset();
        m_ray_traced_shadows.reset();
        m_ray_traced_ao.reset();
        m_ray_traced_reflections.reset();
        m_ddgi.reset();
        m_common_resources.reset();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_pressed(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W)
            m_heading_speed = m_camera_speed;
        else if (code == GLFW_KEY_S)
            m_heading_speed = -m_camera_speed;

        // Handle sideways movement.
        if (code == GLFW_KEY_A)
            m_sideways_speed = -m_camera_speed;
        else if (code == GLFW_KEY_D)
            m_sideways_speed = m_camera_speed;

        if (code == GLFW_KEY_SPACE)
            m_mouse_look = true;

        if (code == GLFW_KEY_G)
            m_debug_gui = !m_debug_gui;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_released(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W || code == GLFW_KEY_S)
            m_heading_speed = 0.0f;

        // Handle sideways movement.
        if (code == GLFW_KEY_A || code == GLFW_KEY_D)
            m_sideways_speed = 0.0f;

        if (code == GLFW_KEY_SPACE)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_pressed(int code) override
    {
        // Enable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_released(int code) override
    {
        // Disable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    dw::AppSettings intial_app_settings() override
    {
        // Set custom settings here...
        dw::AppSettings settings;

        settings.width       = 1920;
        settings.height      = 1080;
        settings.title       = "Hybrid Rendering (c) Dihara Wijetunga";
        settings.ray_tracing = true;
        settings.resizable   = false;

        return settings;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void window_resized(int width, int height) override
    {
        // Override window resized method to update camera projection.
        m_main_camera->update_projection(60.0f, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE, float(m_width) / float(m_height));

        m_vk_backend->wait_idle();

        write_descriptor_sets();
    }

private:
    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_uniform_buffer()
    {
        m_common_resources->ubo_size = m_vk_backend->aligned_dynamic_ubo_size(sizeof(UBO));
        m_common_resources->ubo      = dw::vk::Buffer::create(m_vk_backend, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, m_common_resources->ubo_size * dw::vk::Backend::kMaxFramesInFlight, VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_descriptor_set_layouts()
    {
        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_common_resources->per_frame_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
            m_common_resources->per_frame_ds_layout->set_name("Per Frame DS Layout");
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_common_resources->blue_noise_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
            m_common_resources->blue_noise_ds_layout->set_name("Blue Noise DS Layout");
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
            desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
            desc.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);

            m_common_resources->skybox_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
            m_common_resources->skybox_ds_layout->set_name("Skybox DS Layout");
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);

            m_common_resources->storage_image_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
            m_common_resources->storage_image_ds_layout->set_name("Storage Image DS Layout");
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_common_resources->combined_sampler_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
            m_common_resources->combined_sampler_ds_layout->set_name("Combined Sampler DS Layout");
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            desc.add_binding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

            m_common_resources->ddgi_read_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_descriptor_sets()
    {
        m_common_resources->per_frame_ds = m_vk_backend->allocate_descriptor_set(m_common_resources->per_frame_ds_layout);

        for (int i = 0; i < 9; i++)
            m_common_resources->blue_noise_ds[i] = m_vk_backend->allocate_descriptor_set(m_common_resources->blue_noise_ds_layout);

        int num_environment_map_images = environment_map_images.size() + 2;

        m_common_resources->skybox_ds.resize(num_environment_map_images);

        for (int i = 0; i < num_environment_map_images; i++)
            m_common_resources->skybox_ds[i] = m_vk_backend->allocate_descriptor_set(m_common_resources->skybox_ds_layout);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void write_descriptor_sets()
    {
        // Per-frame
        {
            std::vector<VkWriteDescriptorSet> write_datas;

            {
                VkDescriptorBufferInfo buffer_info;

                buffer_info.range  = sizeof(UBO);
                buffer_info.offset = 0;
                buffer_info.buffer = m_common_resources->ubo->handle();

                VkWriteDescriptorSet write_data;
                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                write_data.pBufferInfo     = &buffer_info;
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_common_resources->per_frame_ds->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo image_info;

                image_info.sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info.imageView   = m_common_resources->blue_noise_view_1->handle();
                image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write_data;
                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_info;
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_common_resources->per_frame_ds->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo image_info;

                image_info.sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info.imageView   = m_common_resources->blue_noise_view_2->handle();
                image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write_data;
                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_info;
                write_data.dstBinding      = 2;
                write_data.dstSet          = m_common_resources->per_frame_ds->handle();

                write_datas.push_back(write_data);
            }

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // Skybox resources
        int num_environment_map_images = environment_map_images.size() + 2;

        for (int i = 0; i < num_environment_map_images; i++)
        {
            VkDescriptorImageInfo image_info[4];

            image_info[0].sampler = m_vk_backend->bilinear_sampler()->handle();
            if (i == ENVIRONMENT_TYPE_NONE)
                image_info[0].imageView = m_common_resources->blank_cubemap_image_view->handle();
            else if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
                image_info[0].imageView = m_common_resources->sky_environment->hosek_wilkie_sky_model->image_view()->handle();
            else
                image_info[0].imageView = m_common_resources->hdr_environments[i - 2]->image_view->handle();
            image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[1].sampler = m_vk_backend->trilinear_sampler()->handle();
            if (i == ENVIRONMENT_TYPE_NONE)
                image_info[1].imageView = m_common_resources->blank_sh_image_view->handle();
            else if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
                image_info[1].imageView = m_common_resources->sky_environment->cubemap_sh_projection->image_view()->handle();
            else
                image_info[1].imageView = m_common_resources->hdr_environments[i - 2]->cubemap_sh_projection->image_view()->handle();
            image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[2].sampler = m_vk_backend->trilinear_sampler()->handle();
            if (i == ENVIRONMENT_TYPE_NONE)
                image_info[2].imageView = m_common_resources->blank_cubemap_image_view->handle();
            else if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
                image_info[2].imageView = m_common_resources->sky_environment->cubemap_prefilter->image_view()->handle();
            else
                image_info[2].imageView = m_common_resources->hdr_environments[i - 2]->cubemap_prefilter->image_view()->handle();
            image_info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[3].sampler     = m_vk_backend->bilinear_sampler()->handle();
            image_info[3].imageView   = m_common_resources->brdf_preintegrate_lut->image_view()->handle();
            image_info[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data[4];
            DW_ZERO_MEMORY(write_data[0]);
            DW_ZERO_MEMORY(write_data[1]);
            DW_ZERO_MEMORY(write_data[2]);
            DW_ZERO_MEMORY(write_data[3]);

            write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[0].descriptorCount = 1;
            write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[0].pImageInfo      = &image_info[0];
            write_data[0].dstBinding      = 0;
            write_data[0].dstSet          = m_common_resources->skybox_ds[i]->handle();

            write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[1].descriptorCount = 1;
            write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[1].pImageInfo      = &image_info[1];
            write_data[1].dstBinding      = 1;
            write_data[1].dstSet          = m_common_resources->skybox_ds[i]->handle();

            write_data[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[2].descriptorCount = 1;
            write_data[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[2].pImageInfo      = &image_info[2];
            write_data[2].dstBinding      = 2;
            write_data[2].dstSet          = m_common_resources->skybox_ds[i]->handle();

            write_data[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[3].descriptorCount = 1;
            write_data[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[3].pImageInfo      = &image_info[3];
            write_data[3].dstBinding      = 3;
            write_data[3].dstSet          = m_common_resources->skybox_ds[i]->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 4, &write_data[0], 0, nullptr);
        }

        m_common_resources->current_skybox_ds = m_common_resources->skybox_ds[m_common_resources->current_environment_type];

        // Blue Noise
        {
            for (int i = 0; i < 9; i++)
            {
                VkDescriptorImageInfo image_info[2];

                image_info[0].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[0].imageView   = m_common_resources->blue_noise->m_sobol_image_view->handle();
                image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[1].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[1].imageView   = m_common_resources->blue_noise->m_scrambling_ranking_image_view[i]->handle();
                image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write_data[2];
                DW_ZERO_MEMORY(write_data[0]);
                DW_ZERO_MEMORY(write_data[1]);

                write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[0].descriptorCount = 1;
                write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[0].pImageInfo      = &image_info[0];
                write_data[0].dstBinding      = 0;
                write_data[0].dstSet          = m_common_resources->blue_noise_ds[i]->handle();

                write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[1].descriptorCount = 1;
                write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[1].pImageInfo      = &image_info[1];
                write_data[1].dstBinding      = 1;
                write_data[1].dstSet          = m_common_resources->blue_noise_ds[i]->handle();

                vkUpdateDescriptorSets(m_vk_backend->device(), 2, &write_data[0], 0, nullptr);
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_tone_map_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_push_constant_range(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ToneMapPushConstants));
        desc.add_descriptor_set_layout(m_common_resources->combined_sampler_ds_layout);

        m_common_resources->copy_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);
        m_common_resources->copy_pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(m_vk_backend, "shaders/triangle.vert.spv", "shaders/tone_map.frag.spv", m_common_resources->copy_pipeline_layout, m_vk_backend->swapchain_render_pass());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_environment_resources()
    {
        // Create procedural sky
        {
            m_common_resources->sky_environment                         = std::unique_ptr<SkyEnvironment>(new SkyEnvironment());
            m_common_resources->sky_environment->hosek_wilkie_sky_model = std::unique_ptr<dw::HosekWilkieSkyModel>(new dw::HosekWilkieSkyModel(m_vk_backend));
            m_common_resources->sky_environment->cubemap_sh_projection  = std::unique_ptr<dw::CubemapSHProjection>(new dw::CubemapSHProjection(m_vk_backend, m_common_resources->sky_environment->hosek_wilkie_sky_model->image()));
            m_common_resources->sky_environment->cubemap_prefilter      = std::unique_ptr<dw::CubemapPrefiler>(new dw::CubemapPrefiler(m_vk_backend, m_common_resources->sky_environment->hosek_wilkie_sky_model->image()));
        }

        // Create blank SH image
        {
            m_common_resources->blank_sh_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, 9, 1, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
            m_common_resources->blank_sh_image->set_name("Blank SH Projection Image");

            m_common_resources->blank_sh_image_view = dw::vk::ImageView::create(m_vk_backend, m_common_resources->blank_sh_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
            m_common_resources->blank_sh_image_view->set_name("Blank SH Projection Image View");

            std::vector<glm::vec4> sh_data(9);
            std::vector<size_t>    sh_sizes(1);

            for (int i = 0; i < 9; i++)
                sh_data[i] = glm::vec4(0.0f);

            sh_sizes[0] = sizeof(glm::vec4) * 9;

            dw::vk::BatchUploader uploader(m_vk_backend);

            uploader.upload_image_data(m_common_resources->blank_sh_image, sh_data.data(), sh_sizes);

            uploader.submit();
        }

        // Create blank environment map
        {
            m_common_resources->blank_cubemap_image      = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, 2, 2, 1, 1, 6, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED, 0, nullptr, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
            m_common_resources->blank_cubemap_image_view = dw::vk::ImageView::create(m_vk_backend, m_common_resources->blank_cubemap_image, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

            std::vector<glm::vec4> cubemap_data(2 * 2 * 6);
            std::vector<size_t>    cubemap_sizes(6);

            int idx = 0;

            for (int layer = 0; layer < 6; layer++)
            {
                cubemap_sizes[layer] = sizeof(glm::vec4) * 4;

                for (int i = 0; i < 4; i++)
                    cubemap_data[idx++] = glm::vec4(0.0f);
            }

            dw::vk::BatchUploader uploader(m_vk_backend);

            uploader.upload_image_data(m_common_resources->blank_cubemap_image, cubemap_data.data(), cubemap_sizes);

            uploader.submit();
        }

        // Load environment maps
        std::unique_ptr<dw::EquirectangularToCubemap> equirectangular_to_cubemap = std::unique_ptr<dw::EquirectangularToCubemap>(new dw::EquirectangularToCubemap(m_vk_backend, VK_FORMAT_R32G32B32A32_SFLOAT));

        m_common_resources->hdr_environments.resize(environment_map_images.size());

        for (int i = 0; i < environment_map_images.size(); i++)
        {
            std::shared_ptr<HDREnvironment> environment = std::shared_ptr<HDREnvironment>(new HDREnvironment());

            auto input_image = dw::vk::Image::create_from_file(m_vk_backend, environment_map_images[i], true);

            environment->image                 = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, 1024, 1024, 1, 5, 6, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED, 0, nullptr, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
            environment->image_view            = dw::vk::ImageView::create(m_vk_backend, environment->image, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
            environment->cubemap_sh_projection = std::unique_ptr<dw::CubemapSHProjection>(new dw::CubemapSHProjection(m_vk_backend, environment->image));
            environment->cubemap_prefilter     = std::unique_ptr<dw::CubemapPrefiler>(new dw::CubemapPrefiler(m_vk_backend, environment->image));

            equirectangular_to_cubemap->convert(input_image, environment->image);

            auto cmd_buf = m_vk_backend->allocate_graphics_command_buffer(true);

            environment->image->generate_mipmaps(cmd_buf);
            environment->cubemap_sh_projection->update(cmd_buf);
            environment->cubemap_prefilter->update(cmd_buf);

            vkEndCommandBuffer(cmd_buf->handle());

            m_vk_backend->flush_graphics({ cmd_buf });

            m_common_resources->hdr_environments[i] = environment;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_mesh()
    {
        m_common_resources->scenes.reserve(SCENE_TYPE_COUNT);

        {
            std::vector<dw::RayTracedScene::Instance> instances;

            dw::Mesh::Ptr pillar = dw::Mesh::load(m_vk_backend, "mesh/pillar.gltf");

            if (!pillar)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            pillar->initialize_for_ray_tracing(m_vk_backend);

            m_common_resources->meshes.push_back(pillar);

            dw::Mesh::Ptr bunny = dw::Mesh::load(m_vk_backend, "mesh/bunny.gltf");

            if (!bunny)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            bunny->initialize_for_ray_tracing(m_vk_backend);

            m_common_resources->meshes.push_back(bunny);

            dw::Mesh::Ptr ground = dw::Mesh::load(m_vk_backend, "mesh/ground.gltf");

            if (!ground)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            ground->initialize_for_ray_tracing(m_vk_backend);

            m_common_resources->meshes.push_back(ground);

            float segment_length = (ground->max_extents().z - ground->min_extents().z) / (NUM_PILLARS + 1);

            for (uint32_t i = 0; i < NUM_PILLARS; i++)
            {
                dw::RayTracedScene::Instance pillar_instance;

                pillar_instance.mesh      = pillar;
                pillar_instance.transform = glm::mat4(1.0f);

                glm::vec3 pos = glm::vec3(15.0f, 0.0f, ground->min_extents().z + segment_length * (i + 1));

                pillar_instance.transform = glm::translate(pillar_instance.transform, pos);

                instances.push_back(pillar_instance);
            }

            for (uint32_t i = 0; i < NUM_PILLARS; i++)
            {
                dw::RayTracedScene::Instance pillar_instance;

                pillar_instance.mesh      = pillar;
                pillar_instance.transform = glm::mat4(1.0f);

                glm::vec3 pos = glm::vec3(-15.0f, 0.0f, ground->min_extents().z + segment_length * (i + 1));

                pillar_instance.transform = glm::translate(pillar_instance.transform, pos);

                instances.push_back(pillar_instance);
            }

            dw::RayTracedScene::Instance ground_instance;

            ground_instance.mesh      = ground;
            ground_instance.transform = glm::mat4(1.0f);

            instances.push_back(ground_instance);

            dw::RayTracedScene::Instance bunny_instance;

            bunny_instance.mesh = bunny;

            glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f));
            glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(135.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.0f));

            bunny_instance.transform = T * R * S;

            instances.push_back(bunny_instance);

            m_common_resources->scenes.push_back(dw::RayTracedScene::create(m_vk_backend, instances));
        }

        {
            std::vector<dw::RayTracedScene::Instance> instances;

            dw::Mesh::Ptr sponza = dw::Mesh::load(m_vk_backend, "mesh/sponza.obj");

            if (!sponza)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            sponza->initialize_for_ray_tracing(m_vk_backend);

            m_common_resources->meshes.push_back(sponza);

            dw::RayTracedScene::Instance sponza_instance;

            sponza_instance.mesh      = sponza;
            sponza_instance.transform = glm::scale(glm::mat4(1.0f), glm::vec3(0.3f));

            instances.push_back(sponza_instance);

            m_common_resources->scenes.push_back(dw::RayTracedScene::create(m_vk_backend, instances));
        }

        {
            std::vector<dw::RayTracedScene::Instance> instances;

            dw::Mesh::Ptr pica_pica = dw::Mesh::load(m_vk_backend, "scene.gltf");

            if (!pica_pica)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            pica_pica->initialize_for_ray_tracing(m_vk_backend);

            m_common_resources->meshes.push_back(pica_pica);

            dw::RayTracedScene::Instance pica_pica_instance;

            pica_pica_instance.mesh      = pica_pica;
            pica_pica_instance.transform = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));

            instances.push_back(pica_pica_instance);

            m_common_resources->scenes.push_back(dw::RayTracedScene::create(m_vk_backend, instances));
        }

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_camera()
    {
        m_main_camera                     = std::make_unique<dw::Camera>(60.0f, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE, float(m_width) / float(m_height), glm::vec3(0.0f, 35.0f, 125.0f), glm::vec3(0.0f, 0.0, -1.0f));
        m_common_resources->prev_position = m_main_camera->m_position;

        float z_buffer_params_x             = -1.0 + (CAMERA_NEAR_PLANE / CAMERA_FAR_PLANE);
        m_common_resources->z_buffer_params = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / CAMERA_NEAR_PLANE, 1.0f / CAMERA_NEAR_PLANE);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void tone_map(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Tone Map", cmd_buf);

        VkClearValue clear_values[2];

        clear_values[0].color.float32[0] = 0.0f;
        clear_values[0].color.float32[1] = 0.0f;
        clear_values[0].color.float32[2] = 0.0f;
        clear_values[0].color.float32[3] = 1.0f;

        clear_values[1].color.float32[0] = 1.0f;
        clear_values[1].color.float32[1] = 1.0f;
        clear_values[1].color.float32[2] = 1.0f;
        clear_values[1].color.float32[3] = 1.0f;

        VkRenderPassBeginInfo info    = {};
        info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass               = m_vk_backend->swapchain_render_pass()->handle();
        info.framebuffer              = m_vk_backend->swapchain_framebuffer()->handle();
        info.renderArea.extent.width  = m_width;
        info.renderArea.extent.height = m_height;
        info.clearValueCount          = 2;
        info.pClearValues             = &clear_values[0];

        vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp;

        vp.x        = 0.0f;
        vp.y        = (float)m_height;
        vp.width    = (float)m_width;
        vp.height   = -(float)m_height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

        VkRect2D scissor_rect;

        scissor_rect.extent.width  = m_width;
        scissor_rect.extent.height = m_height;
        scissor_rect.offset.x      = 0;
        scissor_rect.offset.y      = 0;

        vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_common_resources->copy_pipeline->handle());

        VkDescriptorSet read_ds;

        if (m_temporal_aa->enabled())
            read_ds = m_temporal_aa->output_ds()->handle();
        else
        {
            if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_FINAL)
                read_ds = m_deferred_shading->output_ds()->handle();
            else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_SHADOWS)
                read_ds = m_ray_traced_shadows->output_ds()->handle();
            else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_AMBIENT_OCCLUSION)
                read_ds = m_ray_traced_ao->output_ds()->handle();
            else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_REFLECTIONS)
                read_ds = m_ray_traced_reflections->output_ds()->handle();
            else
                read_ds = m_ddgi->output_ds()->handle();
        }

        VkDescriptorSet descriptor_sets[] = {
            read_ds
        };

        ToneMapPushConstants push_constants;

        push_constants.single_channel = (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_SHADOWS || m_common_resources->current_visualization_type == VISUALIZATION_TYPE_AMBIENT_OCCLUSION) ? 1 : 0;
        push_constants.exposure       = m_exposure;

        vkCmdPushConstants(cmd_buf->handle(), m_common_resources->copy_pipeline_layout->handle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ToneMapPushConstants), &push_constants);

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_common_resources->copy_pipeline_layout->handle(), 0, 1, descriptor_sets, 0, nullptr);

        vkCmdDraw(cmd_buf->handle(), 3, 1, 0, 0);

        render_gui(cmd_buf);

        vkCmdEndRenderPass(cmd_buf->handle());
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

        m_ubo_data.proj_inverse        = glm::inverse(m_common_resources->projection);
        m_ubo_data.view_inverse        = glm::inverse(m_common_resources->view);
        m_ubo_data.view_proj           = m_common_resources->projection * m_common_resources->view;
        m_ubo_data.view_proj_inverse   = glm::inverse(m_ubo_data.view_proj);
        m_ubo_data.prev_view_proj      = m_common_resources->first_frame ? m_common_resources->prev_view_projection : current_jitter * m_common_resources->prev_view_projection;
        m_ubo_data.cam_pos             = glm::vec4(m_common_resources->position, float(m_deferred_shading->use_ray_traced_ao()));
        m_ubo_data.current_prev_jitter = glm::vec4(m_temporal_aa->current_jitter(), m_temporal_aa->prev_jitter());

        set_light_radius(m_ubo_data.light, m_light_radius);
        set_light_direction(m_ubo_data.light, m_light_direction);
        set_light_color(m_ubo_data.light, m_light_color);
        set_light_intensity(m_ubo_data.light, m_light_intensity);
        set_light_type(m_ubo_data.light, LIGHT_TYPE_DIRECTIONAL);

        m_main_camera->m_prev_view_projection = m_ubo_data.view_proj;

        uint8_t* ptr = (uint8_t*)m_common_resources->ubo->mapped_ptr();
        memcpy(ptr + m_common_resources->ubo_size * m_vk_backend->current_frame_idx(), &m_ubo_data, sizeof(UBO));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_ibl(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        if (m_common_resources->current_environment_type == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
        {
            m_common_resources->sky_environment->hosek_wilkie_sky_model->update(cmd_buf, m_light_direction);

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
            double time = glfwGetTime() * 0.5f;

            m_light_direction.x = sinf(time);
            m_light_direction.z = cosf(time);
            m_light_direction.y = 1.0f;
            m_light_direction   = glm::normalize(m_light_direction);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_camera()
    {
        m_temporal_aa->update();

        dw::Camera* current = m_main_camera.get();

        float forward_delta = m_heading_speed * m_delta;
        float right_delta   = m_sideways_speed * m_delta;

        current->set_translation_delta(current->m_forward, forward_delta);
        current->set_translation_delta(current->m_right, right_delta);

        m_camera_x = m_mouse_delta_x * m_camera_sensitivity;
        m_camera_y = m_mouse_delta_y * m_camera_sensitivity;

        if (m_mouse_look)
        {
            // Activate Mouse Look
            current->set_rotatation_delta(glm::vec3((float)(m_camera_y),
                                                    (float)(m_camera_x),
                                                    (float)(0.0f)));
        }
        else
        {
            current->set_rotatation_delta(glm::vec3((float)(0),
                                                    (float)(0),
                                                    (float)(0)));
        }

        current->update();

        m_common_resources->frame_time    = m_delta_seconds;
        m_common_resources->camera_delta  = m_main_camera->m_position - m_common_resources->prev_position;
        m_common_resources->prev_position = m_main_camera->m_position;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void set_active_scene()
    {
        if (m_common_resources->current_scene_type == SCENE_TYPE_PILLARS)
        {
            m_ddgi->set_normal_bias(1.0f);
            m_ddgi->set_probe_distance(4.0f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(0.5f);
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_SPONZA)
        {
            m_ddgi->set_normal_bias(0.1f);
            m_ddgi->set_probe_distance(50.0f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(5.0f);
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_PICA_PICA)
        {
            m_ddgi->set_normal_bias(1.0f);
            m_ddgi->set_probe_distance(4.0f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(0.5f);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    std::unique_ptr<CommonResources>      m_common_resources;
    std::unique_ptr<GBuffer>              m_g_buffer;
    std::unique_ptr<DeferredShading>      m_deferred_shading;
    std::unique_ptr<RayTracedShadows>     m_ray_traced_shadows;
    std::unique_ptr<RayTracedAO>          m_ray_traced_ao;
    std::unique_ptr<RayTracedReflections> m_ray_traced_reflections;
    std::unique_ptr<DDGI>                 m_ddgi;
    std::unique_ptr<TemporalAA>           m_temporal_aa;

    // Camera.
    std::unique_ptr<dw::Camera> m_main_camera;
    bool                        m_mouse_look         = false;
    float                       m_heading_speed      = 0.0f;
    float                       m_sideways_speed     = 0.0f;
    float                       m_camera_sensitivity = 0.05f;
    float                       m_camera_speed       = 0.2f;
    float                       m_offset             = 0.1f;
    bool                        m_debug_gui          = false;

    // Camera orientation.
    float m_camera_x;
    float m_camera_y;

    // Light
    float     m_light_radius    = 0.1f;
    glm::vec3 m_light_direction = glm::normalize(glm::vec3(0.568f, 0.707f, -0.421f));
    glm::vec3 m_light_color     = glm::vec3(1.0f);
    float     m_light_intensity = 1.0f;
    bool      m_light_animation = false;

    // Uniforms.
    UBO   m_ubo_data;
    float m_exposure = 1.0f;
};

DW_DECLARE_MAIN(HybridRendering)