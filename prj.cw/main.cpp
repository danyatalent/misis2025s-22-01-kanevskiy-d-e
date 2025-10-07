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


// Загружаем JSON и извлекаем 4 точки ROI
std::vector<cv::Point2f> loadPolygonROIFromJson(const std::string& json_path) {
    std::ifstream in(json_path);
    if (!in.is_open()) {
        throw std::runtime_error("Не удалось открыть JSON: " + json_path);
    }

    json j;
    in >> j;

    std::vector<cv::Point2f> polygon;
    for (const auto& pt : j["points"]) {
        float x = pt["x"];
        float y = pt["y"];
        polygon.emplace_back(x, y);
    }

    if (polygon.size() != 4) {
        throw std::runtime_error("Ожидалось 4 точки в JSON: " + json_path);
    }

    return polygon;
}

cv::Mat cropAndAlignByPolygon(const cv::Mat& img, const std::vector<cv::Point2f>& polygon) {
    if (polygon.size() != 4) {
        throw std::invalid_argument("polygon должен содержать ровно 4 точки");
    }

    // Вычисляем ширину и высоту выровненного изображения
    const float width_bottom = cv::norm(polygon[1] - polygon[0]);
    const float width_top    = cv::norm(polygon[2] - polygon[3]);
    float width = std::max(width_bottom, width_top);

    const float height_left  = cv::norm(polygon[3] - polygon[0]);
    const float height_right = cv::norm(polygon[2] - polygon[1]);
    float height = std::max(height_left, height_right);

    // Целевые точки: прямой прямоугольник
    const std::vector<cv::Point2f> dst_pts = {
        {0.f, height},       // левый нижний
        {width, height},     // правый нижний
        {width, 0.f},        // правый верхний
        {0.f, 0.f}           // левый верхний
    };

    // Матрица трансформации и применение
    const cv::Mat M = cv::getPerspectiveTransform(polygon, dst_pts);
    cv::Mat aligned;
    cv::warpPerspective(img, aligned, M, cv::Size(static_cast<int>(width), static_cast<int>(height)));

    return aligned;
}



int main(const int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: main_cw <image_path_lst> <json_path_lst> <output_path_lst> <input_rate(1/k)> <tmp_path>" << std::endl;
        return -1;
    }

    const fs::path image_path_lst = argv[1];
    const fs::path json_path_lst = argv[2];
    const fs::path output_path_lst = argv[3];
    const std::string input_rate = argv[4];
    const fs::path tmp_path = argv[5];

    auto image_paths = get_list_of_file_paths(image_path_lst);
    auto json_paths = get_list_of_file_paths(json_path_lst);
    auto output_paths = get_list_of_file_paths(output_path_lst);
    auto tmp_paths = get_list_of_file_paths(tmp_path);

    std::ofstream timings_file("timings.csv"); // создаёт файл при запуске
    if (!timings_file.is_open()) {
        std::cerr << "Failed to open timings file for writing." << std::endl;
        return -1;
    }
    timings_file << "filename,k,duration_sec\n";

    for (int i = 0; i < image_paths.size(); i++)
    {
        // Загружаем изображение и json
        const cv::Mat img = cv::imread(image_paths[i], cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Image not found: " << image_paths[i] << std::endl;
            return -1;
        }

        // Делаем кроп
        // Загружаем 4 точки
        std::vector<cv::Point2f> roi_pts = loadPolygonROIFromJson(json_paths[i]);

        // Получаем выровненный кроп
        cv::Mat img_crop = cropAndAlignByPolygon(img, roi_pts);

        const clock_t start = clock();

        // Удаляем тень
        const cv::Mat result = removeShadowWaterFilling(img_crop, std::stof(input_rate), tmp_paths[i]);
        const int input_k = 1/std::stof(input_rate);

        const double duration = (clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
        std::cout << "time: " << duration  << " sec" << std::endl;
        timings_file << image_paths[i].filename() << ","
                 << input_k << ","
                 << duration << "\n";
        // Сохраняем
        cv::imwrite(output_paths[i], result);
    }
    return 0;
}
