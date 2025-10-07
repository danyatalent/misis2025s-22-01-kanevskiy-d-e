//
// Created by danya on 20.06.2025.
//

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <opencv2/ximgproc/edge_filter.hpp>

namespace fs = std::filesystem;

#ifndef WATER_FILLING_H
cv::Mat water_filling(const cv::Mat& src, cv::Size original_size, const fs::path& path);
cv::Mat incre_filling(cv::Mat input, cv::Mat Original, const fs::path& path);
cv::Mat removeShadowWaterFilling(const cv::Mat& input, float rate, const fs::path& path);
#define WATER_FILLING_H

#endif //WATER_FILLING_H
