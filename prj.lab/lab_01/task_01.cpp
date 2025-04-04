//
// Created by danya on 03.04.2025.
//
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "semcv/semcv.hpp"

int main(const int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_lst_file>" << std::endl;
        return 1;
    }

    const fs::path lst_path = argv[1];


    try {
        for (const auto file_paths = get_list_of_file_paths(lst_path); const auto& file_path : file_paths) {
            cv::Mat img = cv::imread(file_path.string(), cv::IMREAD_UNCHANGED);

            if (img.empty()) {
                std::cout << file_path.filename().string() << "\tbad, should be UNREADABLE" << std::endl;
                continue;
            }

            std::string expected_format = strid_from_mat(img);

            std::string file_name = fs::path(file_path).filename().string();

            if (const size_t dot_pos = file_name.rfind('.'); dot_pos != std::string::npos) {
                file_name = file_name.substr(0, dot_pos);
            }

            if (file_name == expected_format) {
                std::cout << file_name << "\tgood" << std::endl;
            } else {
                std::cout << file_name << "\tbad, should be " << expected_format << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}