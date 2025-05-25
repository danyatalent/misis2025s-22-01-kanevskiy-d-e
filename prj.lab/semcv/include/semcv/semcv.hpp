#ifndef SEMCV_HPP_
#define SEMCV_HPP_

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace semcv
{
    std::string strid_from_mat(const cv::Mat& img, int n = 4);
    std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst);
    cv::Mat generate_gray_stripes_mat();
    cv::Mat gamma_correction(const cv::Mat& img, double gamma);

    struct DistributionStats {
        double mean;
        double stddev;
    };

    cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2);
    cv::Mat add_noise_gau(const cv::Mat& img, const int std);
    DistributionStats compute_stats(const cv::Mat& img, const cv::Mat& mask);
    void create_masks(const int size, const int square_side, const int circle_radius,
                  cv::Mat& mask_bg, cv::Mat& mask_square, cv::Mat& mask_circle);
    cv::Mat draw_histogram(const cv::Mat& img_input, const cv::Scalar& bg_color = cv::Scalar(220));
    cv::Mat make_histogram_grid(const std::vector<cv::Mat>& images);
}


#endif
