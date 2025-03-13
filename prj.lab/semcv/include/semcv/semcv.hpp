#ifndef SEMCV_HPP_
#define SEMCV_HPP_

#include <opencv2/opencv.hpp>
#include <filesystem>

std::string strid_from_mat(const cv::Mat& img, int n = 4);

std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst);
#endif
