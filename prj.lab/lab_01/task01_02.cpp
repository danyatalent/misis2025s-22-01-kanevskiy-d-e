#include <opencv2/opencv.hpp>
#include <iostream>
#include "semcv/semcv.hpp"

int main(const int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_path>" << std::endl;
        return 1;
    }

    const std::string output_path = argv[1];

    const cv::Mat striped_img = semcv::generate_gray_stripes_mat();

    std::vector gammas = { 1.8, 2.0, 2.2, 2.4, 2.6 };
    std::vector<cv::Mat> gamma_images;

    for (const double gamma : gammas) {
        gamma_images.push_back(semcv::gamma_correction(striped_img, gamma));
    }

    cv::Mat output_collage;
    vconcat(striped_img, gamma_images[0], output_collage);
    for (size_t i = 1; i < gamma_images.size(); ++i) {
        vconcat(output_collage, gamma_images[i], output_collage);
    }

    if (imwrite(output_path, output_collage)) {
        std::cout << "Collage saved to: " << output_path << std::endl;
    }
    else {
        std::cerr << "Error: Failed to save collage to " << output_path << std::endl;
    }

    return 0;
}