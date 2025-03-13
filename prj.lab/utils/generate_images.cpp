//
// Created by danya on 21.02.2025.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

template <typename T>
void fillImage(Mat& img, int depth) {
    randu(img, Scalar::all(0), Scalar::all(256));
}

template <>
void fillImage<float>(Mat& img, int depth) {
    randu(img, Scalar::all(0.0f), Scalar::all(1.0f));
}

template <>
void fillImage<double>(Mat& img, int depth) {
    randu(img, Scalar::all(0.0), Scalar::all(1.0));
}

struct ImageConfig {
    int type;
    int channels;
    int depth;
    string typeName;
};

vector<ImageConfig> getImageConfigs() {
    return {
        {CV_8U,  1, CV_8U,  "CV_8UC1"},
        {CV_8U,  3, CV_8U,  "CV_8UC3"},
        {CV_8U,  4, CV_8U,  "CV_8UC4"},
        {CV_16U, 1, CV_16U, "CV_16UC1"},
        {CV_16U, 3, CV_16U, "CV_16UC3"},
        {CV_32F, 1, CV_32F, "CV_32FC1"},
        {CV_32F, 3, CV_32F, "CV_32FC3"},
        {CV_64F, 1, CV_64F, "CV_64FC1"}
    };
}

bool saveImage(const string& path, const Mat& img, const string& format) {

    if (format == "JPEG"  && (img.channels() == 4 || img.depth() != CV_8U))
    {
        return false;
    }

    return imwrite(path, img);
}

int main() {
    const string output_dir = "test_images_all";
    const Size image_size(256, 256);

    vector<string> formats = {"JPEG", "PNG", "TIFF"};
    auto configs = getImageConfigs();

    try {
        fs::create_directories(output_dir);

    }
    catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
        return -1;
    }
    std::ofstream outfile("test_images_all/task01.lst");

    RNG rng(12345);
    int total = 0, success = 0;

    for (const auto& cfg : configs) {
        Mat img(image_size, CV_MAKETYPE(cfg.depth, cfg.channels));

        switch (cfg.depth) {
            case CV_8U:  fillImage<uchar>(img, cfg.depth); break;
            case CV_16U: fillImage<ushort>(img, cfg.depth); break;
            case CV_32F: fillImage<float>(img, cfg.depth); break;
            case CV_64F: fillImage<double>(img, cfg.depth); break;
            default: continue;
        }

        for (const auto& fmt : formats) {
            string ext = "." + fmt;
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            string filename = cfg.typeName + "_" + fmt + ext;

            string path = output_dir + "/" + filename;

            total++;
            if (saveImage(path, img, fmt)) {
                success++;
                outfile << filename << endl;
                cout << "Saved: " << path << endl;
            }
        }
    }

    outfile.close();

    cout << "\nTotal processed: " << total
         << "\nSuccessfully saved: " << success
         << "\nFailed: " << (total - success) << endl;

    return 0;
}