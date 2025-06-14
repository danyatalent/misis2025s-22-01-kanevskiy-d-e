#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct DetectedObject {
    float angle;
    float width;
    float height;
    int x;
    int y;
};

void saveDetectionsToJson(const std::string& filename, const std::vector<DetectedObject>& detections) {
    json result;
    result["objects"] = json::array();

    for (const auto& detection : detections)
    {
        json object;
        object["elps_parameters"]["elps_x"] = detection.x;
        object["elps_parameters"]["elps_y"] = detection.y;
        object["elps_parameters"]["elps_width"] = detection.width;
        object["elps_parameters"]["elps_height"] = detection.height;
        object["elps_parameters"]["elps_angle"] = detection.angle;

        result["objects"].push_back(object);
    }

    std::ofstream out_file(filename);
    if (!out_file) {
        std::cerr << "Error: Could not open output file\n";
        return;
    }

    out_file << std::setw(4) << result << std::endl;
}

std::vector<cv::KeyPoint> detectBlobs(const cv::Mat& img) {
    cv::Mat processed;
    cv::GaussianBlur(img, processed, cv::Size(9, 9), 0);
    cv::normalize(processed, processed, 0, 255, cv::NORM_MINMAX, CV_32F);
    cv::morphologyEx(processed, processed, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    std::vector<cv::KeyPoint> key_points;
    constexpr int num_levels = 3;
    constexpr float scale_factor = 2.0f;
    constexpr float min_response = 0.04f * 255;

    std::vector<cv::Mat> pyramid;
    pyramid.push_back(processed);
    for (int i = 1; i < num_levels; ++i) {
        cv::Mat scaled;
        auto scale = static_cast<float>(std::pow(scale_factor, i));
        cv::resize(processed, scaled, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_LINEAR);
        pyramid.push_back(scaled);
    }

    for (int level = 0; level < num_levels; ++level) {
        constexpr int num_scales = 3;
        auto scale = static_cast<float>(std::pow(scale_factor, level));
        cv::Mat prev_gaussian;

        for (int i = 0; i < num_scales; ++i) {
            constexpr float k = 1.414f;
            constexpr float sigma = 5.0f;
            auto current_sigma = static_cast<float>(sigma * std::pow(k, i));
            cv::Mat gaussian;
            cv::GaussianBlur(pyramid[level], gaussian, cv::Size(0, 0), current_sigma, current_sigma, cv::BORDER_REPLICATE);

            if (i > 0) {
                cv::Mat dog;
                cv::subtract(gaussian, prev_gaussian, dog, cv::noArray(), CV_32F);

                cv::Mat dog_abs;
                cv::absdiff(dog, cv::Scalar(0), dog_abs);

                cv::Mat dilated;
                cv::dilate(dog_abs, dilated, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
                cv::Mat local_max = (dog_abs >= dilated) & (dog_abs > min_response);

                std::vector<cv::Point> points;
                cv::findNonZero(local_max, points);

                for (const auto& p : points) {
                    auto blob_size = static_cast<float>(current_sigma * scale * std::sqrt(2));
                    if (constexpr float min_diameter = 50.0f; blob_size < min_diameter) continue;

                    cv::KeyPoint kp;
                    kp.pt = cv::Point2f(p.x * scale, p.y * scale);
                    kp.size = blob_size;
                    kp.response = dog_abs.at<float>(p);

                    bool is_duplicate = false;
                    for (const auto& existing_kp : key_points) {
                        float dist = static_cast<float>(cv::norm(kp.pt - existing_kp.pt));
                        if (constexpr float overlap_threshold = 0.7f; dist < std::min(kp.size, existing_kp.size) * overlap_threshold) {
                            is_duplicate = true;
                            break;
                        }
                    }

                    if (!is_duplicate) {
                        key_points.push_back(kp);
                    }
                }
            }
            prev_gaussian = gaussian.clone();
        }
    }

    std::vector<cv::KeyPoint> filtered_keypoints;
    float response_threshold = min_response * 2.0f;
    for (const auto& kp : key_points) {
        if (kp.response > response_threshold) {
            filtered_keypoints.push_back(kp);
        }
    }

    return filtered_keypoints;
}

std::vector<DetectedObject> detectEllipses(const cv::Mat& image) {
    std::vector<DetectedObject> detections;

    for (std::vector<cv::KeyPoint> key_points = detectBlobs(image); const auto& kp : key_points) {
        const float size = kp.size * 2.0f;
        const cv::RotatedRect ellipse(kp.pt, cv::Size2f(size, size), 0.0f);

        if (const double area = ellipse.size.width * ellipse.size.height * CV_PI / 4.0; area < 500 || area > 100000) continue;

        DetectedObject obj{};
        obj.x = static_cast<int>(ellipse.center.x);
        obj.y = static_cast<int>(ellipse.center.y);
        obj.width = ellipse.size.width;
        obj.height = ellipse.size.height;
        obj.angle = ellipse.angle;

        detections.push_back(obj);
    }

    return detections;
}

int main(const int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: task06 <image_path> <output_json>" << std::endl;
        return -1;
    }

    const std::string imagePath = argv[1];
    const std::string outputJson = argv[2];

    const cv::Mat imageGray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (imageGray.empty()) {
        std::cerr << "Error loading image! Check file path: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat imageColor = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (imageColor.empty()) {
        std::cerr << "Error loading color image! Check file path: " << imagePath << std::endl;
        return -1;
    }

    std::vector<DetectedObject> detections = detectEllipses(imageGray);
    saveDetectionsToJson(outputJson, detections);

    for (const DetectedObject& obj : detections) {
        const cv::Point center(obj.x, obj.y);
        const int radius = static_cast<int>((obj.width + obj.height) / 4.0);
        cv::Scalar color(0, 0, 255);
        cv::circle(imageColor, center, radius, color, 2);
    }

    std::filesystem::path jsonPath(outputJson);
    const std::string outputImagePath = jsonPath.replace_extension(".png").string();
    cv::imwrite(outputImagePath, imageColor);

    std::cout << "Detection results saved to " << outputJson << std::endl;
    std::cout << "Visualization saved to " << outputImagePath << std::endl;

    return 0;
}
