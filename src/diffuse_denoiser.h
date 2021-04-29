#pragma once

#include <vk.h>
#include <glm.hpp>

class CommonResources;
class GBuffer;
class TemporalReprojection;
class BilateralBlur;

class DiffuseDenoiser
{
public:
    DiffuseDenoiser(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height);
    ~DiffuseDenoiser();

    void                       denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

private:
    std::string      m_name;
    CommonResources* m_common_resources;
    uint32_t         m_input_width;
    uint32_t         m_input_height;
    bool             m_use_blur_as_temporal_input = false;

    // Temporal
    std::unique_ptr<TemporalReprojection> m_temporal_reprojection;

    // Bilateral Blur
    std::unique_ptr<BilateralBlur> m_bilateral_blur;
};
