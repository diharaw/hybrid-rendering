#include "diffuse_denoiser.h"
#include "temporal_reprojection.h"
#include "bilateral_blur.h"
#include "g_buffer.h"
#include "common_resources.h"
#include "utilities.h"
#include <macros.h>
#include <profiler.h>
#include <imgui.h>

DiffuseDenoiser::DiffuseDenoiser(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_common_resources(common_resources), m_input_width(input_width), m_input_height(input_height)
{
    m_temporal_reprojection = std::unique_ptr<TemporalReprojection>(new TemporalReprojection(backend, common_resources, g_buffer, name, input_width, input_height));
    m_bilateral_blur        = std::unique_ptr<BilateralBlur>(new BilateralBlur(backend, common_resources, g_buffer, name, input_width, input_height));

    m_temporal_reprojection->set_variance_clipping(false);
    m_temporal_reprojection->set_neighborhood_scale(3.5f);
    m_temporal_reprojection->set_alpha(0.01f);
    m_bilateral_blur->set_blur_radius(5);
}

DiffuseDenoiser::~DiffuseDenoiser()
{
}

void DiffuseDenoiser::denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    if (m_common_resources->first_frame)
        m_bilateral_blur->prepare_first_frame(cmd_buf);

    m_temporal_reprojection->reproject(cmd_buf, input, m_use_blur_as_temporal_input ? m_bilateral_blur->output_ds() : nullptr);
    m_bilateral_blur->blur(cmd_buf, m_temporal_reprojection->output_ds());
}

void DiffuseDenoiser::gui()
{
    ImGui::Checkbox("Use Blur as Temporal Input", &m_use_blur_as_temporal_input);
    m_temporal_reprojection->gui();
    m_bilateral_blur->gui();
}

dw::vk::DescriptorSet::Ptr DiffuseDenoiser::output_ds()
{
    return m_bilateral_blur->output_ds();
}
