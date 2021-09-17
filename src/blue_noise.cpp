#include "blue_noise.h"

// -----------------------------------------------------------------------------------------------------------------------------------

static const char* kSOBOL_TEXTURE = "textures/blue_noise/sobol_256_4d.png";

// -----------------------------------------------------------------------------------------------------------------------------------

static const char* kSCRAMBLING_RANKING_TEXTURES[] = {
    "textures/blue_noise/scrambling_ranking_128x128_2d_1spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_2spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_4spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_8spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_16spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_32spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_64spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_128spp.png",
    "textures/blue_noise/scrambling_ranking_128x128_2d_256spp.png"
};

// -----------------------------------------------------------------------------------------------------------------------------------

BlueNoise::BlueNoise(dw::vk::Backend::Ptr backend)
{
    m_sobol_image      = dw::vk::Image::create_from_file(backend, kSOBOL_TEXTURE);
    m_sobol_image_view = dw::vk::ImageView::create(backend, m_sobol_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

    for (int i = 0; i < 9; i++)
    {
        m_scrambling_ranking_image[i]      = dw::vk::Image::create_from_file(backend, kSCRAMBLING_RANKING_TEXTURES[i]);
        m_scrambling_ranking_image_view[i] = dw::vk::ImageView::create(backend, m_scrambling_ranking_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

BlueNoise::~BlueNoise()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------