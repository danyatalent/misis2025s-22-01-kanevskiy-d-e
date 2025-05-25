//
// Created by danya on 25.05.2025.
//
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "semcv/semcv.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: task02 <output_path> <hist_path>\n";
        return 1;
    }

    std::string output_path = argv[1];
    std::string hist_path = argv[2];
    std::vector<std::tuple<int, int, int>> levels = {
        {0, 127, 255}, {20, 127, 235}, {55, 127, 200}, {90, 127, 165}
    };

    std::vector<cv::Mat> originals;
    for (const auto& [lev0, lev1, lev2] : levels) {
        originals.push_back(semcv::gen_tgtimg00(lev0, lev1, lev2));
    }

    cv::Mat original_concat;
    cv::hconcat(originals, original_concat);

    std::vector stds = {3, 7, 15};
    std::vector all_rows = {original_concat};

    std::vector<cv::Mat> all_images = originals;

    for (int s : stds) {
        cv::Mat noisy = semcv::add_noise_gau(original_concat, s);
        all_rows.push_back(noisy);
        for (int i = 0; i + 256 <= noisy.cols; i += 256) {
            all_images.push_back(noisy.colRange(i, i + 256).clone());
        }
    }

    cv::Mat final_image;
    cv::vconcat(all_rows, final_image);

    if (!cv::imwrite(output_path, final_image)) {
        std::cerr << "Failed to save image to: " << output_path << std::endl;
        return 1;
    }

    std::cout << "Image saved to: " << output_path << std::endl;

    // Statistics output
    cv::Mat mask_bg, mask_sq, mask_circ;
    semcv::create_masks(256, 209, 83, mask_bg, mask_sq, mask_circ);

    for (size_t i = 0; i < all_images.size(); ++i) {
        auto& img = all_images[i];
        auto bg_stat = semcv::compute_stats(img, mask_bg);
        auto sq_stat = semcv::compute_stats(img, mask_sq);
        auto cr_stat = semcv::compute_stats(img, mask_circ);

        std::cout << "Image " << i << ":\n";
        std::cout << "  BG    : mean=" << bg_stat.mean << " std=" << bg_stat.stddev << "\n";
        std::cout << "  Square: mean=" << sq_stat.mean << " std=" << sq_stat.stddev << "\n";
        std::cout << "  Circle: mean=" << cr_stat.mean << " std=" << cr_stat.stddev << "\n";
    }

    // Histogram visualization
    cv::Mat hist_image = semcv::make_histogram_grid(all_images);
    if (hist_image.empty()) {
        std::cerr << "Histogram image is empty!" << std::endl;
    } else {
        std::cout << "Histogram image size: " << hist_image.size()
                  << ", type: " << hist_image.type() << std::endl;
    }
    if (!cv::imwrite(hist_path, hist_image)) {
        std::cerr << "Error code: " << cv::getBuildInformation() << std::endl;
        std::cerr << "Check path and permissions: " << hist_path << std::endl;
    }

    return 0;
}