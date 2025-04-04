#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

string getDepthName(const int depth) {
    switch (depth) {
        case CV_8U:  return "uint08";
        case CV_16U: return "uint16";
        case CV_32F: return "real32";
        default:     return "unknown";
    }
}

void saveImage(const Mat& img, const string& filepath, const string& format) {
    vector<int> compression_params;

    if (format == "jpeg") {
        compression_params.push_back(IMWRITE_JPEG_QUALITY);
        compression_params.push_back(95);
    } else if (format == "png") {
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(3);
    } else if (format == "tiff") {
        compression_params.push_back(IMWRITE_TIFF_COMPRESSION);
        compression_params.push_back(1);
    }

    imwrite(filepath, img, compression_params);
    cout << "Saved: " << filepath << endl;
}

Mat generateImage(const int rows, const int cols, const int type) {
    Mat img(rows, cols, type);

    return img;
}

int main() {
    vector<int> depths = {CV_8U, CV_16U, CV_32F};
    vector<int> channels = {1, 3, 4};
    vector<string> formats = {"png", "jpeg", "tiff"};
    int index = 0;

    string folderName = "images";
    if (!fs::exists(folderName)) {
        fs::create_directory(folderName);
    }

    ofstream lstFile(folderName + "/task01.lst");
    if (!lstFile.is_open()) {
        cerr << "Не удалось создать файл task01.lst" << endl;
        return -1;
    }

    for (int depth : depths) {
        for (int ch : channels) {
            for (const string& format : formats) {
                Mat img = generateImage(256, 256, CV_MAKETYPE(depth, ch));

                string depthName = getDepthName(depth);
                string filename = "0256x0256." + to_string(ch) + "." + depthName + "." + format;
                string filepath = folderName + "/" + filename;

                saveImage(img, filepath, format);

                lstFile << filename << endl;
            }
        }
    }

    lstFile.close();
    cout << "Все изображения и файл task01.lst успешно сохранены в папке images/!" << endl;

    return 0;
}
