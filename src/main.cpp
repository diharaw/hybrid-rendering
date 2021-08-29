#include <application.h>
#include <camera.h>
#include <profiler.h>
#include <assimp/scene.h>
#include <equirectangular_to_cubemap.h>
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
#include "utilities.h"

#define EPSILON 0.0001f
#define NUM_PILLARS 6
#define CAMERA_NEAR_PLANE 1.0f
#define CAMERA_FAR_PLANE 1000.0f
#define CAMERA_SPEED_MULTIPLIER 0.1f

const std::vector<std::string>            environment_map_images        = { "textures/Arches_E_PineTree_3k.hdr", "textures/BasketballCourt_3k.hdr", "textures/Etnies_Park_Center_3k.hdr", "textures/LA_Downtown_Helipad_GoldenHour_3k.hdr" };
const std::vector<std::string>            environment_types             = { "None", "Procedural Sky", "Arches Pine Tree", "Basketball Court", "Etnies Park Central", "LA Downtown Helipad" };
const std::vector<std::string>            visualization_types           = { "Final", "Shadows", "Ambient Occlusion", "Reflections", "Global Illumination", "Ground Truth" };
const std::vector<std::string>            scene_types                   = { "Pillars", "Reflections Test", "Sponza", "Pica Pica" };
const std::vector<std::string>            ray_trace_scales              = { "Full-Res", "Half-Res", "Quarter-Res" };
const std::vector<std::string>            light_types                   = { "Directional", "Point", "Spot" };
const std::vector<std::string>            camera_types                  = { "Free", "Animated", "Fixed" };
const std::vector<std::vector<glm::vec3>> fixed_camera_position_vectors = {
    { glm::vec3(-22.061460f, 16.624475f, 23.893597f),
      glm::vec3(-0.337131f, 15.421529f, 39.524925f),
      glm::vec3(9.907501f, 8.313064f, -18.302629f),
      glm::vec3(10.431265f, 4.411600f, -6.578662f) },

    { glm::vec3(-42.599087f, 5.077470f, 3.662686f),
      glm::vec3(42.569202f, 5.680231f, 1.135333f),
      glm::vec3(-0.088592f, 18.299492f, 31.712112f),
      glm::vec3(-10.971692f, 4.040000f, -1.099626f) },

    { glm::vec3(349.689911f, 50.013187f, -47.142761f),
      glm::vec3(255.940018f, 181.126541f, -14.534848f),
      glm::vec3(25.954714f, 36.763203f, 153.194244f),
      glm::vec3(-391.294556f, 179.648758f, 141.655914f) },

    { glm::vec3(-2.880592f, 12.838152f, 28.133095f),
      glm::vec3(-4.044456f, 3.885819f, 14.471013f),
      glm::vec3(-10.408246f, 4.111171f, 8.519235f),
      glm::vec3(-10.283543f, 6.659785f, 2.117568f) }
};
const std::vector<std::vector<glm::vec3>> fixed_camera_forward_vectors = {
    { glm::vec3(0.593151f, -0.521760f, -0.613138f),
      glm::vec3(-0.006306f, -0.425798f, -0.904796f),
      glm::vec3(-0.353051f, -0.351048f, 0.867249f),
      glm::vec3(-0.800752f, -0.151261f, 0.579584f) },

    { glm::vec3(0.926363f, -0.233447f, -0.295558f),
      glm::vec3(-0.956285f, -0.235149f, -0.173853f),
      glm::vec3(0.003158f, -0.457108f, -0.889406f),
      glm::vec3(-0.593590f, -0.287377f, -0.751709f) },

    { glm::vec3(-0.927807f, -0.008728f, 0.372960f),
      glm::vec3(-0.890209f, -0.455542f, -0.003118f),
      glm::vec3(0.932927f, -0.008722f, -0.359960f),
      glm::vec3(0.723851f, -0.095842f, -0.683267f) },

    { glm::vec3(-0.005560f, -0.393157f, -0.919454f),
      glm::vec3(0.725216f, -0.146966f, -0.672653f),
      glm::vec3(-0.739586f, -0.270623f, -0.616259f),
      glm::vec3(0.787594f, -0.314029f, -0.530172f) }
};
const std::vector<std::vector<glm::vec3>> fixed_camera_right_vectors = {
    { glm::vec3(0.718724f, -0.000000f, 0.695295f),
      glm::vec3(0.999976f, 0.000000f, -0.006970f),
      glm::vec3(-0.926194f, 0.000000f, -0.377048f),
      glm::vec3(-0.586330f, 0.000000f, -0.810072f) },

    { glm::vec3(0.303957f, -0.000000f, 0.952686f),
      glm::vec3(0.178869f, 0.000000f, -0.983873f),
      glm::vec3(0.999994f, -0.000000f, 0.003551f),
      glm::vec3(0.784814f, 0.000000f, -0.619732f) },

    { glm::vec3(-0.372974f, 0.000000f, -0.927842f),
      glm::vec3(0.003502f, 0.000000f, -0.999994f),
      glm::vec3(0.359974f, -0.000000f, 0.932963f),
      glm::vec3(0.686427f, -0.000000f, 0.727199f) },

    { glm::vec3(0.999982f, 0.000000f, -0.006047f),
      glm::vec3(0.680037f, -0.000000f, 0.733178f),
      glm::vec3(0.640146f, 0.000000f, -0.768253f),
      glm::vec3(0.558420f, -0.000000f, 0.829558f) }
};

struct Light
{
    glm::vec4 data0;
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
};

void set_light_direction(Light& light, glm::vec3 value)
{
    light.data0.x = value.x;
    light.data0.y = value.y;
    light.data0.z = value.z;
}

void set_light_position(Light& light, glm::vec3 value)
{
    light.data1.x = value.x;
    light.data1.y = value.y;
    light.data1.z = value.z;
}

void set_light_color(Light& light, glm::vec3 value)
{
    light.data2.x = value.x;
    light.data2.y = value.y;
    light.data2.z = value.z;
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
    light.data3.x = value;
}

void set_light_cos_theta_outer(Light& light, float value)
{
    light.data3.y = value;
}

void set_light_cos_theta_inner(Light& light, float value)
{
    light.data3.z = value;
}

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
        m_common_resources->blue_noise            = std::unique_ptr<BlueNoise>(new BlueNoise(m_vk_backend));
        m_common_resources->blue_noise_image_1    = dw::vk::Image::create_from_file(m_vk_backend, "texture/LDR_RGBA_0.png");
        m_common_resources->blue_noise_view_1     = dw::vk::ImageView::create(m_vk_backend, m_common_resources->blue_noise_image_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_common_resources->blue_noise_image_2    = dw::vk::Image::create_from_file(m_vk_backend, "texture/LDR_RGBA_1.png");
        m_common_resources->blue_noise_view_2     = dw::vk::ImageView::create(m_vk_backend, m_common_resources->blue_noise_image_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

        create_environment_resources();
        create_descriptor_set_layouts();
        create_descriptor_sets();
        write_descriptor_sets();

        m_g_buffer                 = std::unique_ptr<GBuffer>(new GBuffer(m_vk_backend, m_common_resources.get(), m_width, m_height));
        m_ray_traced_shadows       = std::unique_ptr<RayTracedShadows>(new RayTracedShadows(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ray_traced_ao            = std::unique_ptr<RayTracedAO>(new RayTracedAO(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ray_traced_reflections   = std::unique_ptr<RayTracedReflections>(new RayTracedReflections(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ddgi                     = std::unique_ptr<DDGI>(new DDGI(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_ground_truth_path_tracer = std::unique_ptr<GroundTruthPathTracer>(new GroundTruthPathTracer(m_vk_backend, m_common_resources.get()));
        m_deferred_shading         = std::unique_ptr<DeferredShading>(new DeferredShading(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_temporal_aa              = std::unique_ptr<TemporalAA>(new TemporalAA(m_vk_backend, m_common_resources.get(), m_g_buffer.get()));
        m_tone_map                 = std::unique_ptr<ToneMap>(new ToneMap(m_vk_backend, m_common_resources.get()));

        set_active_scene();
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

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
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

            dw::Mesh::Ptr reflections_test = dw::Mesh::load(m_vk_backend, "mesh/reflections_test.gltf");

            if (!reflections_test)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            reflections_test->initialize_for_ray_tracing(m_vk_backend);

            m_common_resources->meshes.push_back(reflections_test);

            dw::RayTracedScene::Instance reflections_test_instance;

            reflections_test_instance.mesh      = reflections_test;
            reflections_test_instance.transform = glm::mat4(1.0f);

            instances.push_back(reflections_test_instance);

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
            ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(m_width * 0.3f, m_height));

            if (ImGui::Begin("Hybrid Rendering", &open, window_flags))
            {
                if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::TreeNode("General"))
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

                                if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY && m_light_type != LIGHT_TYPE_DIRECTIONAL)
                                    continue;

                                if (ImGui::Selectable(environment_types[i].c_str(), is_selected))
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
                        else if (m_common_resources->current_visualization_type == VISUALIZATION_TYPE_GROUND_TRUTH)
                            m_ground_truth_path_tracer->gui();

                        m_tone_map->gui();

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Light"))
                    {
                        LightType type = m_light_type;

                        if (ImGui::BeginCombo("Type", light_types[type].c_str()))
                        {
                            for (uint32_t i = 0; i < light_types.size(); i++)
                            {
                                const bool is_selected = (i == type);

                                if (ImGui::Selectable(light_types[i].c_str(), is_selected))
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

                        if (ImGui::BeginCombo("Type", camera_types[type].c_str()))
                        {
                            for (uint32_t i = 0; i < camera_types.size(); i++)
                            {
                                const bool is_selected = (i == type);

                                if (ImGui::Selectable(camera_types[i].c_str(), is_selected))
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
                                for (uint32_t i = 0; i < fixed_camera_forward_vectors[m_common_resources->current_scene_type].size(); i++)
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

                        ImGui::SliderFloat("Speed", &m_camera_speed, 0.1f, 10.0f);

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Ray Traced Shadows"))
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

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Ray Traced Reflections"))
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

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Ray Traced Ambient Occlusion"))
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

                        ImGui::TreePop();
                        ImGui::Separator();
                    }
                    if (ImGui::TreeNode("Global Illumination"))
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

        ImGui::Checkbox("Animation", &m_light_animation);
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

        ImGui::Checkbox("Animation", &m_light_animation);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void reset_light()
    {
        m_light_transform = glm::mat4(1.0f);

        if (m_common_resources->current_scene_type == SCENE_TYPE_PILLARS)
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

        set_light_radius(m_ubo_data.light, m_light_radius);
        set_light_color(m_ubo_data.light, m_light_color);
        set_light_intensity(m_ubo_data.light, m_light_intensity);
        set_light_type(m_ubo_data.light, m_light_type);
        set_light_direction(m_ubo_data.light, -m_light_direction);
        set_light_position(m_ubo_data.light, m_light_position);
        set_light_cos_theta_inner(m_ubo_data.light, glm::cos(glm::radians(m_light_cone_angle_inner)));
        set_light_cos_theta_outer(m_ubo_data.light, glm::cos(glm::radians(m_light_cone_angle_outer)));

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

        if (m_camera_type == CAMERA_TYPE_FREE)
        {
            float forward_delta = m_heading_speed * m_delta;
            float sideways_delta   = m_sideways_speed * m_delta;

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

            m_main_camera->update();
        }
        else if (m_camera_type == CAMERA_TYPE_FIXED)
            m_main_camera->update_from_frame(fixed_camera_position_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle], fixed_camera_forward_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle], fixed_camera_right_vectors[m_common_resources->current_scene_type][m_current_fixed_camera_angle]);

        m_common_resources->frame_time    = m_delta_seconds;
        m_common_resources->camera_delta  = m_main_camera->m_position - m_common_resources->prev_position;
        m_common_resources->prev_position = m_main_camera->m_position;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void set_active_scene()
    {
        m_current_fixed_camera_angle = 0;

        if (m_common_resources->current_scene_type == SCENE_TYPE_PILLARS)
        {
            m_ddgi->set_normal_bias(1.0f);
            m_ddgi->set_probe_distance(4.0f);
            m_ddgi->restart_accumulation();
            m_deferred_shading->set_probe_visualization_scale(0.5f);
        }
        else if (m_common_resources->current_scene_type == SCENE_TYPE_REFLECTIONS_TEST)
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
    bool                        m_mouse_look         = false;
    float                       m_heading_speed      = 0.0f;
    float                       m_sideways_speed     = 0.0f;
    float                       m_camera_sensitivity = 0.05f;
    float                       m_camera_speed       = 2.0f;
    float                       m_offset             = 0.1f;
    bool                        m_debug_gui          = false;

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
    bool                m_light_animation           = false;
    LightType           m_light_type                = LIGHT_TYPE_DIRECTIONAL;

    // Uniforms.
    UBO m_ubo_data;
};

DW_DECLARE_MAIN(HybridRendering)