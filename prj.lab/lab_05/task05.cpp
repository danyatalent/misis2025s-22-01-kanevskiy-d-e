#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <test_image_path> <result_image_path>\n";
            return 1;
        }

        const std::string test_image_path = argv[1];
        const std::string result_image_path = argv[2];

        constexpr int square_size = 127;
        constexpr int width = 3 * square_size;
        constexpr int height = 2 * square_size;

        cv::Mat test_image(height, width, CV_8UC1, cv::Scalar(0));

        const std::vector<std::pair<uchar, uchar>> combinations = {
            {0, 127}, {0, 255}, {127, 0},
            {127, 255}, {255, 0}, {255, 127}
        };

        int idx = 0;
        for (int y = 0; y < height; y += square_size) {
            for (int x = 0; x < width; x += square_size) {
                constexpr int circle_radius = 40;
                if (idx >= combinations.size()) break;

                uchar square_intensity = combinations[idx].first;
                uchar circle_intensity = combinations[idx].second;
                idx++;

                cv::rectangle(test_image,
                              cv::Rect(x, y, square_size, square_size),
                              square_intensity,
                              cv::FILLED);

                cv::circle(test_image,
                           cv::Point(x + square_size/2, y + square_size/2),
                           circle_radius,
                           circle_intensity,
                           cv::FILLED);
            }
        }

        cv::imwrite(test_image_path, test_image);

        cv::Mat I1, I2, I3;

        cv::Mat kernel1 = (cv::Mat_<float>(3, 3) <<
            1, 0, -1,
            2, 0, -2,
            1, 0, -1);
        cv::filter2D(test_image, I1, CV_32F, kernel1);

        cv::Mat kernel2 = (cv::Mat_<float>(3, 3) <<
            -1, -2, -1,
            0, 0, 0,
            1, 2, 1);
        cv::filter2D(test_image, I2, CV_32F, kernel2);

        cv::magnitude(I1, I2, I3);

        I1.convertTo(I1, CV_8U, 1.0, 128);
        I2.convertTo(I2, CV_8U, 1.0, 128);
        I3.convertTo(I3, CV_8U, 1.0, 128);

        cv::Mat V4;
        std::vector<cv::Mat> channels = {I1, I2, I3};
        cv::merge(channels, V4);

        cv::Mat visualization(2 * height, 2 * width, CV_8UC3, cv::Scalar(128, 128, 128));

        cv::Mat color_I1, color_I2, color_I3;
        cv::cvtColor(I1, color_I1, cv::COLOR_GRAY2BGR);
        cv::cvtColor(I2, color_I2, cv::COLOR_GRAY2BGR);
        cv::cvtColor(I3, color_I3, cv::COLOR_GRAY2BGR);

        color_I1.copyTo(visualization(cv::Rect(0, 0, width, height)));
        color_I2.copyTo(visualization(cv::Rect(width, 0, width, height)));
        color_I3.copyTo(visualization(cv::Rect(0, height, width, height)));
        V4.copyTo(visualization(cv::Rect(width, height, width, height)));

        cv::imwrite(result_image_path, visualization);
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}