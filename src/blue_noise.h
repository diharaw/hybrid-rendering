#pragma once

#include <vk.h>

enum BlueNoiseSpp
{
    BLUE_NOISE_1SPP,
    BLUE_NOISE_2SPP,
    BLUE_NOISE_4SPP,
    BLUE_NOISE_8SPP,
    BLUE_NOISE_16SPP,
    BLUE_NOISE_32SPP,
    BLUE_NOISE_64SPP,
    BLUE_NOISE_128SPP
};

struct BlueNoise
{
    dw::vk::Image::Ptr m_sobol_image;
    dw::vk::Image::Ptr m_scrambling_ranking_image[9];

    dw::vk::ImageView::Ptr m_sobol_image_view;
    dw::vk::ImageView::Ptr m_scrambling_ranking_image_view[9];

    BlueNoise(dw::vk::Backend::Ptr backend);
    ~BlueNoise();
};