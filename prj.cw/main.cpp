#include "water_filling.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

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

int main(const int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: main_cw <image_path_lst> <json_path_lst> <output_path_lst> <input_rate(1/k)>" << std::endl;
        return -1;
    }

    const fs::path image_path_lst = argv[1];
    const fs::path json_path_lst = argv[2];
    const fs::path output_path_lst = argv[3];
    const std::string input_rate = argv[4];

    auto image_paths = get_list_of_file_paths(image_path_lst);
    auto json_paths = get_list_of_file_paths(json_path_lst);
    auto output_paths = get_list_of_file_paths(output_path_lst);

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
        const cv::Rect roi = loadROIFromJson(json_paths[i]);
        const cv::Mat img_crop = img(roi).clone();

        const clock_t start = clock();

        // Удаляем тень
        const cv::Mat result = removeShadowWaterFilling(img_crop, std::stof(input_rate));
        const int input_k = 1/std::stof(input_rate);

        const double duration = (clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
        std::cout << "time: " << duration  << " sec" << std::endl;
        timings_file << image_paths[i].filename() << ","
                 << input_k << ","
                 << duration << "\n";
        // Сохраняем
        cv::imwrite(output_paths[i], result);
    }
    // // Загружаем изображение и json
    // const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    // if (img.empty()) {
    //     std::cerr << "Image not found: " << image_path << std::endl;
    //     return -1;
    // }
    //
    // // Делаем кроп
    // const cv::Rect roi = loadROIFromJson(json_path);
    // const cv::Mat img_crop = img(roi).clone();
    // // cv::imwrite(crop_path, img_crop);
    //
    // const clock_t start = clock();
    //
    // // Удаляем тень
    // const cv::Mat result = removeShadowWaterFilling(img_crop);
    //
    // const double duration = (clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
    // std::cout << "time: " << duration  << " sec" << std::endl;
    // // Сохраняем
    // cv::imwrite(output_path_1, result);
    return 0;
}
