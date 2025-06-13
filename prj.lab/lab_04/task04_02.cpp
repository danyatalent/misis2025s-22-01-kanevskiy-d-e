#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <output_json_path>\n";
        return 1;
    }

    std::string image_path = argv[1];
    std::string output_path = argv[2];

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // 1. Бинаризация - простая пороговая обработка
    cv::Mat binary;
    cv::threshold(img, binary, 50, 255, cv::THRESH_BINARY);

    // 2. Морфологические операции - только закрытие для заполнения мелких отверстий
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    // 3. Поиск контуров
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    json result;
    result["objects"] = json::array();

    for (const auto & contour : contours) {
        double area = cv::contourArea(contour);
        if (constexpr double min_contour_area = 100.0; area < min_contour_area) continue;

        if (contour.size() < 5) continue;

        cv::RotatedRect ellipse = cv::fitEllipse(contour);

        if (ellipse.size.width < 10 || ellipse.size.width > 200) continue;
        if (ellipse.size.height < 10 || ellipse.size.height > 200) continue;

        json obj;
        obj["elps_parameters"]["elps_x"] = ellipse.center.x;
        obj["elps_parameters"]["elps_y"] = ellipse.center.y;
        obj["elps_parameters"]["elps_width"] = ellipse.size.width;
        obj["elps_parameters"]["elps_height"] = ellipse.size.height;
        obj["elps_parameters"]["elps_angle"] = ellipse.angle;

        result["objects"].push_back(obj);
    }

    // Сохранение результатов
    std::ofstream out_file(output_path);
    if (!out_file) {
        std::cerr << "Could not open output file: " << output_path << std::endl;
        return 1;
    }
    out_file << std::setw(4) << result << std::endl;

    cv::Mat debug_img;
    cv::cvtColor(img, debug_img, cv::COLOR_GRAY2BGR);
    for (auto& obj : result["objects"]) {
        auto params = obj["elps_parameters"];
        cv::RotatedRect ellipse(
            cv::Point2f(params["elps_x"], params["elps_y"]),
            cv::Size2f(params["elps_width"], params["elps_height"]),
            params["elps_angle"]
        );
        cv::ellipse(debug_img, ellipse, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("debug_detection.jpg", debug_img);

    std::cout << "Detection completed successfully. Found "
              << result["objects"].size() << " objects." << std::endl;
    std::cout << "Debug visualization saved to debug_detection.jpg" << std::endl;

    return 0;
}