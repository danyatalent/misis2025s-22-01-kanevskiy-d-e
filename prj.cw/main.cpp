#include "water_filling.h"

using json = nlohmann::json;

// Загружаем JSON и извлекаем ROI
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

int main(const int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: main_cw <image_path> <json_path> <output_path_1>" << std::endl;
        return -1;
    }

    const std::string image_path = argv[1];
    const std::string json_path = argv[2];
    const std::string output_path_1 = argv[3];

    // Загружаем изображение и json
    const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Image not found: " << image_path << std::endl;
        return -1;
    }

    // Делаем кроп
    const cv::Rect roi = loadROIFromJson(json_path);
    const cv::Mat img_crop = img(roi).clone();
    // cv::imwrite(crop_path, img_crop);

    const clock_t start = clock();

    // Удаляем тень
    const cv::Mat result = removeShadowWaterFilling(img_crop);

    const double duration = (clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
    std::cout << "time: " << duration  << " sec" << std::endl;
    // Сохраняем
    cv::imwrite(output_path_1, result);
    return 0;
}
