#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

struct EllipseParams {
    float x;
    float y;
    float width;
    float height;
    float angle;
};

struct ImageResult {
    int gt_count;
    int detected_count;
    int matched_count;
    double avg_iou;
    double avg_center_dist;
    double avg_size_diff;
    double avg_angle_diff;
};

std::vector<std::string> read_list_file(const std::string& file_path) {
    std::vector<std::string> files;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open list file: " + file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            files.push_back(line);
        }
    }
    return files;
}

double center_distance(const EllipseParams& gt, const EllipseParams& det) {
    return std::sqrt(std::pow(gt.x - det.x, 2) + std::pow(gt.y - det.y, 2));
}

double size_difference(const EllipseParams& gt, const EllipseParams& det) {
    double width_diff = std::abs(gt.width - det.width);
    double height_diff = std::abs(gt.height - det.height);
    return (width_diff + height_diff) / 2.0;
}

double angle_difference(const double gt_angle, const double det_angle) {
    double diff = std::abs(gt_angle - det_angle);
    return std::min(diff, 360.0 - diff);
}

double calculate_iou(const EllipseParams& gt, const EllipseParams& det, const int collage_size) {
    cv::Mat mask_gt = cv::Mat::zeros(collage_size, collage_size, CV_8UC1);
    cv::Mat mask_det = cv::Mat::zeros(collage_size, collage_size, CV_8UC1);

    const cv::RotatedRect rect_gt(cv::Point2f(gt.x, gt.y),
                     cv::Size2f(gt.width, gt.height),
                     gt.angle);
    cv::ellipse(mask_gt, rect_gt, cv::Scalar(255), -1);

    const cv::RotatedRect rect_det(cv::Point2f(det.x, det.y),
                      cv::Size2f(det.width, det.height),
                      det.angle);
    cv::ellipse(mask_det, rect_det, cv::Scalar(255), -1);

    cv::Mat intersection, uni;
    cv::bitwise_and(mask_gt, mask_det, intersection);
    cv::bitwise_or(mask_gt, mask_det, uni);

    const double area_intersection = cv::countNonZero(intersection);
    const double area_union = cv::countNonZero(uni);

    if (area_union < 1e-5) return 0.0;
    return area_intersection / area_union;
}

ImageResult compare_gt_and_det(const json& gt_data, const json& det_data) {
    ImageResult result = {0, 0, 0, 0, 0, 0, 0};

    const int n = gt_data["size_of_collage"];
    const int collage_size = n * 256;

    std::vector<EllipseParams> gt_objects;
    for (const auto& obj : gt_data["objects"]) {
        const int col = obj["pic_coordinates"]["col"];
        const int row = obj["pic_coordinates"]["row"];

        EllipseParams gt{};
        gt.x = col * 256 + obj["elps_parameters"]["elps_x"].get<float>();
        gt.y = row * 256 + obj["elps_parameters"]["elps_y"].get<float>();
        gt.width = obj["elps_parameters"]["elps_width"].get<float>();
        gt.height = obj["elps_parameters"]["elps_height"].get<float>();
        gt.angle = obj["elps_parameters"]["elps_angle"].get<float>();

        gt_objects.push_back(gt);
    }

    std::vector<EllipseParams> det_objects;
    for (const auto& obj : det_data["objects"]) {
        EllipseParams det{};
        det.x = obj["elps_parameters"]["elps_x"].get<float>();
        det.y = obj["elps_parameters"]["elps_y"].get<float>();
        det.width = obj["elps_parameters"]["elps_width"].get<float>();
        det.height = obj["elps_parameters"]["elps_height"].get<float>();
        det.angle = obj["elps_parameters"]["elps_angle"].get<float>();

        det_objects.push_back(det);
    }

    result.gt_count = gt_objects.size();
    result.detected_count = det_objects.size();

    std::vector<bool> matched_det(det_objects.size(), false);
    std::vector<int> matched_gt(gt_objects.size(), -1);

    for (size_t i = 0; i < gt_objects.size(); i++) {
        double min_dist = std::numeric_limits<double>::max();
        int best_match = -1;

        for (size_t j = 0; j < det_objects.size(); j++) {
            if (matched_det[j]) continue;

            const double dist = center_distance(gt_objects[i], det_objects[j]);
            if (constexpr double MAX_DIST = 100.0; dist < min_dist && dist < MAX_DIST) {
                min_dist = dist;
                best_match = j;
            }
        }

        if (best_match != -1) {
            matched_det[best_match] = true;
            matched_gt[i] = best_match;
            result.matched_count++;

            const double iou = calculate_iou(gt_objects[i], det_objects[best_match], collage_size);
            const double center_dist = min_dist;
            const double size_diff = size_difference(gt_objects[i], det_objects[best_match]);
            const double angle_diff = angle_difference(gt_objects[i].angle, det_objects[best_match].angle);

            result.avg_iou += iou;
            result.avg_center_dist += center_dist;
            result.avg_size_diff += size_diff;
            result.avg_angle_diff += angle_diff;
        }
    }

    if (result.matched_count > 0) {
        result.avg_iou /= result.matched_count;
        result.avg_center_dist /= result.matched_count;
        result.avg_size_diff /= result.matched_count;
        result.avg_angle_diff /= result.matched_count;
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <gt_list> <detect_list> <protocol_path>\n";
        return 1;
    }

    std::string gt_list_path = argv[1];
    std::string detect_list_path = argv[2];
    std::string protocol_path = argv[3];

    try {
        auto gt_files = read_list_file(gt_list_path);
        auto detect_files = read_list_file(detect_list_path);

        if (gt_files.size() != detect_files.size()) {
            throw std::runtime_error("Number of GT files (" + std::to_string(gt_files.size()) +
                                    ") does not match number of detection files (" +
                                    std::to_string(detect_files.size()) + ")");
        }

        double total_avg_iou = 0.0;
        double total_avg_center_dist = 0.0;
        double total_avg_size_diff = 0.0;
        double total_avg_angle_diff = 0.0;
        int total_gt_count = 0;
        int total_detected_count = 0;
        int total_matched_count = 0;
        int valid_comparisons = 0;

        std::ofstream protocol(protocol_path);
        if (!protocol) {
            throw std::runtime_error("Could not open protocol file: " + protocol_path);
        }

        protocol << "Quality Assessment Report\n";
        protocol << "=========================\n\n";
        protocol << "Comparing " << gt_files.size() << " images\n\n";
        protocol << "Image\tGT\tDet\tMatch\tIoU\t\tCenter Dist\tSize Diff\tAngle Diff\n";
        protocol << "---------------------------------------------------------------------\n";

        for (size_t i = 0; i < gt_files.size(); i++) {
            try {
                std::ifstream gt_file(gt_files[i]);
                std::ifstream det_file(detect_files[i]);

                if (!gt_file || !det_file) {
                    protocol << "Image " << i+1 << "\tERROR: Could not open files\n";
                    continue;
                }

                json gt_data, det_data;
                try {
                    gt_file >> gt_data;
                    det_file >> det_data;
                } catch (const std::exception& e) {
                    protocol << "Image " << i+1 << "\tERROR: JSON parse error - " << e.what() << "\n";
                    continue;
                }

                ImageResult res = compare_gt_and_det(gt_data, det_data);

                total_gt_count += res.gt_count;
                total_detected_count += res.detected_count;
                total_matched_count += res.matched_count;

                if (res.matched_count > 0) {
                    total_avg_iou += res.avg_iou;
                    total_avg_center_dist += res.avg_center_dist;
                    total_avg_size_diff += res.avg_size_diff;
                    total_avg_angle_diff += res.avg_angle_diff;
                    valid_comparisons++;
                }

                protocol << i+1 << "\t"
                         << res.gt_count << "\t"
                         << res.detected_count << "\t"
                         << res.matched_count << "\t"
                         << std::fixed << std::setprecision(3)
                         << res.avg_iou << "\t\t"
                         << res.avg_center_dist << "\t\t"
                         << res.avg_size_diff << "\t\t"
                         << res.avg_angle_diff << "\n";

            } catch (const std::exception& e) {
                protocol << "Image " << i+1 << "\tERROR: " << e.what() << "\n";
            }
        }

        double overall_avg_iou = valid_comparisons > 0 ? total_avg_iou / valid_comparisons : 0.0;
        double overall_avg_center_dist = valid_comparisons > 0 ? total_avg_center_dist / valid_comparisons : 0.0;
        double overall_avg_size_diff = valid_comparisons > 0 ? total_avg_size_diff / valid_comparisons : 0.0;
        double overall_avg_angle_diff = valid_comparisons > 0 ? total_avg_angle_diff / valid_comparisons : 0.0;

        double precision = total_detected_count > 0 ?
            static_cast<double>(total_matched_count) / total_detected_count : 0.0;
        double recall = total_gt_count > 0 ?
            static_cast<double>(total_matched_count) / total_gt_count : 0.0;
        double f1 = (precision + recall) > 0 ?
            2 * precision * recall / (precision + recall) : 0.0;

        protocol << "\nSummary:\n";
        protocol << "---------------------------------------------------------------------\n";
        protocol << "Total GT objects: " << total_gt_count << "\n";
        protocol << "Total detected objects: " << total_detected_count << "\n";
        protocol << "Total matched objects: " << total_matched_count << "\n";
        protocol << "Precision: " << std::setprecision(4) << precision << "\n";
        protocol << "Recall: " << recall << "\n";
        protocol << "F1 Score: " << f1 << "\n";
        protocol << "Average IoU: " << std::setprecision(3) << overall_avg_iou << "\n";
        protocol << "Average Center Distance: " << overall_avg_center_dist << "\n";
        protocol << "Average Size Difference: " << overall_avg_size_diff << "\n";
        protocol << "Average Angle Difference: " << overall_avg_angle_diff << "\n";
        
        std::cout << "Report generated successfully: " << protocol_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}