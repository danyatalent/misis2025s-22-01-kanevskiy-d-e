#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Config {
    int n;
    int bg_color;
    int elps_color;
    double noise_std;
    int blur_size;
    int min_elps_width;
    int max_elps_width;
    int min_elps_height;
    int max_elps_height;
};

void from_json(const json& j, Config& config) {
    j.at("n").get_to(config.n);
    j.at("bg_color").get_to(config.bg_color);
    j.at("elps_color").get_to(config.elps_color);
    j.at("noise_std").get_to(config.noise_std);
    j.at("blur_size").get_to(config.blur_size);
    j.at("min_elps_width").get_to(config.min_elps_width);
    j.at("max_elps_width").get_to(config.max_elps_width);
    j.at("min_elps_height").get_to(config.min_elps_height);
    j.at("max_elps_height").get_to(config.max_elps_height);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_path> [<output_image_path> <output_gt_path> [seed]]\n";
        return 1;
    }

    std::string config_path = argv[1];

    if (argc == 2) {
        json default_config;
        default_config["n"] = 5;
        default_config["bg_color"] = 0;
        default_config["elps_color"] = 255;
        default_config["noise_std"] = 15.0;
        default_config["blur_size"] = 25;
        default_config["min_elps_width"] = 20;
        default_config["max_elps_width"] = 160;
        default_config["min_elps_height"] = 10;
        default_config["max_elps_height"] = 150;

        std::ofstream config_file(config_path);
        if (!config_file) {
            std::cerr << "Cannot open config file for writing: " << config_path << std::endl;
            return 1;
        }
        config_file << std::setw(4) << default_config << std::endl;
        std::cout << "Default config generated at: " << config_path << std::endl;
        return 0;
    }

    if (argc < 4) {
        std::cerr << "Not enough arguments: expected at least 3, got " << argc-1 << std::endl;
        return 1;
    }

    std::string image_path = argv[2];
    std::string gt_path = argv[3];

    std::ifstream config_file(config_path);
    if (!config_file) {
        std::cerr << "Cannot open config file: " << config_path << std::endl;
        return 1;
    }

    json config_json;
    try {
        config_file >> config_json;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config file: " << e.what() << std::endl;
        return 1;
    }

    Config config{};
    try {
        config = config_json.get<Config>();
    } catch (const std::exception& e) {
        std::cerr << "Error reading config: " << e.what() << std::endl;
        return 1;
    }

    unsigned int seed;
    if (argc >= 5) {
        seed = std::stoul(argv[4]);
    } else
    {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 gen(seed);

    int collage_size = config.n * 256;
    cv::Mat collage(collage_size, collage_size, CV_8UC1, config.bg_color);

    constexpr int margin = 32;
    constexpr int safe_zone = 256 - 2 * margin;

    std::uniform_int_distribution<int> x_dist(margin, 256 - margin);
    std::uniform_int_distribution<int> y_dist(margin, 256 - margin);
    std::uniform_int_distribution<int> width_dist(config.min_elps_width, config.max_elps_width);
    std::uniform_int_distribution<int> height_dist(config.min_elps_height, config.max_elps_height);
    std::uniform_real_distribution<double> angle_dist(0.0, 360.0);

    json gt_json;
    gt_json["blur_size"] = config.blur_size;
    gt_json["colors"]["bg_color"] = config.bg_color;
    gt_json["colors"]["elps_color"] = config.elps_color;
    gt_json["noise_std"] = config.noise_std;
    gt_json["size_of_collage"] = config.n;
    gt_json["objects"] = json::array();

    for (int i = 0; i < config.n; ++i) {
        for (int j = 0; j < config.n; ++j) {
            cv::Rect roi(j * 256, i * 256, 256, 256);
            cv::Mat tile = collage(roi);

            bool valid = false;
            int x, y, width, height;
            double angle;
            int attempts = 0;
            constexpr int max_attempts = 100;

            while (!valid && attempts++ < max_attempts) {
                x = x_dist(gen);
                y = y_dist(gen);
                width = width_dist(gen);
                height = height_dist(gen);
                angle = angle_dist(gen);

                double a = width / 2.0;
                double b = height / 2.0;
                double max_radius = std::max(a, b);

                double rad_angle = angle * CV_PI / 180.0;
                double cos_a = std::abs(std::cos(rad_angle));
                double sin_a = std::abs(std::sin(rad_angle));

                double max_x = a * cos_a + b * sin_a;
                double max_y = a * sin_a + b * cos_a;

                // Проверка границ
                valid = (x - max_x >= margin) &&
                        (x + max_x <= 256 - margin) &&
                        (y - max_y >= margin) &&
                        (y + max_y <= 256 - margin);
            }

            // Если не удалось сгенерировать подходящий эллипс, используем минимальный безопасный
            if (!valid) {
                width = std::min(config.min_elps_width, safe_zone);
                height = std::min(config.min_elps_height, safe_zone);
                x = 128;
                y = 128;
                angle = 0;
            }

            cv::ellipse(tile,
                        cv::Point(x, y),
                        cv::Size(width/2, height/2),
                        angle,
                        0, 360,
                        config.elps_color,
                        -1);

            json obj;
            obj["pic_coordinates"]["row"] = i;
            obj["pic_coordinates"]["col"] = j;
            obj["elps_parameters"]["elps_x"] = x;
            obj["elps_parameters"]["elps_y"] = y;
            obj["elps_parameters"]["elps_width"] = width;
            obj["elps_parameters"]["elps_height"] = height;
            obj["elps_parameters"]["elps_angle"] = angle;

            gt_json["objects"].push_back(obj);
        }
    }

    int blur_size_adj = config.blur_size;
    if (blur_size_adj % 2 == 0) {
        blur_size_adj++;
    }
    cv::GaussianBlur(collage, collage, cv::Size(blur_size_adj, blur_size_adj), 0);

    cv::Mat noise(collage.size(), collage.type());
    cv::randn(noise, cv::Scalar(0), cv::Scalar(config.noise_std));

    cv::Mat collage_16s;
    collage.convertTo(collage_16s, CV_16S);
    cv::Mat noise_16s;
    noise.convertTo(noise_16s, CV_16S);

    collage_16s += noise_16s;

    cv::max(collage_16s, 0, collage_16s);
    cv::min(collage_16s, 255, collage_16s);
    collage_16s.convertTo(collage, CV_8U);

    if (!cv::imwrite(image_path, collage)) {
        std::cerr << "Failed to save image: " << image_path << std::endl;
        return 1;
    }

    std::ofstream gt_file(gt_path);
    if (!gt_file) {
        std::cerr << "Failed to open ground truth file for writing: " << gt_path << std::endl;
        return 1;
    }
    gt_file << std::setw(4) << gt_json << std::endl;

    std::cout << "Collage generated successfully." << std::endl;
    std::cout << "Image saved to: " << image_path << std::endl;
    std::cout << "Ground truth saved to: " << gt_path << std::endl;
    std::cout << "Seed used: " << seed << std::endl;

    return 0;
}