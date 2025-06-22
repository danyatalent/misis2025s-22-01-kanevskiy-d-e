//
// Created by danya on 22.06.2025.
//
#include <opencv4/opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;
namespace fs = std::filesystem;

cv::Rect loadROIFromJson(const std::string& json_path) {
    std::ifstream in(json_path);
    json j;
    in >> j;

    int x = j["x"];
    int y = j["y"];
    int w = j["width"];
    int h = j["height"];

    return {x, y, w, h};
}

std::vector<fs::path> get_list_of_file_paths(const fs::path& path_lst) {
    std::vector<fs::path> file_paths;
    std::ifstream infile(path_lst);
    std::string line;
    fs::path lst_directory = path_lst.parent_path();

    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open lst file: " + path_lst.string());
    }

    while (std::getline(infile, line)) {
        if (!line.empty()) {
            fs::path file_path = lst_directory / line;
            file_paths.push_back(file_path);
        }
    }

    return file_paths;
}

double getMSSIM(const cv::Mat& i1, const cv::Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat I1, I2;
    i1.convertTo(I1, CV_32F);
    i2.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2;
    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);

    cv::Scalar mssim = mean(ssim_map);
    return (mssim[0] + mssim[1] + mssim[2])/3;
}

int main(const int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: psnr <image_path_lst> <gt_path_lst> <gt_json_path_lst>" << std::endl;
        return -1;
    }

    const fs::path image_path_lst = argv[1];
    const fs::path gt_img_path_lst = argv[2];
    const fs::path gt_json_path_lst = argv[3];

    auto image_paths = get_list_of_file_paths(image_path_lst);
    auto json_paths = get_list_of_file_paths(gt_json_path_lst);
    auto gt_paths = get_list_of_file_paths(gt_img_path_lst);

    std::ofstream metrics_file("metrics.csv"); // создаёт файл при запуске
    if (!metrics_file.is_open()) {
        std::cerr << "Failed to open timings file for writing." << std::endl;
        return -1;
    }
    metrics_file << "filename,psnr,ssim\n";

    for (int i = 0; i < image_paths.size(); i++)
    {
        const cv::Mat result = cv::imread(image_paths[i]);
        const cv::Mat gt = cv::imread(gt_paths[i]);
        if (result.empty() || gt.empty()) {
            std::cerr << "Error: could not load images.\n";
            return -1;
        }

        cv::Rect gt_rect = loadROIFromJson(json_paths[i]);
        gt_rect &= cv::Rect(0, 0, gt.cols, gt.rows); // safety clip
        cv::Mat gt_cropped = gt(gt_rect);

        if (gt_cropped.size() != result.size()) {
            std::cout << "resizing\n";
            resize(gt_cropped, gt_cropped, result.size());
        }

        // ===== Стандартные метрики =====
        const double psnr = cv::PSNR(gt_cropped, result);
        const double ssim = getMSSIM(gt_cropped, result);

        metrics_file << image_paths[i].filename() << ","
                 << psnr << ","
                 << ssim << "\n";
    }


    // std::cout << "[RAW] PSNR: " << psnr << " dB\n";
    // std::cout << "[RAW] SSIM: B: " << ssim[0] << " G: " << ssim[1] << " R: " << ssim[2] << "\n";

    return 0;
}
