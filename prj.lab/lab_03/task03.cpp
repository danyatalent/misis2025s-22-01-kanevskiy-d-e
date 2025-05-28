#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "semcv/semcv.hpp"


int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <output_collage_image> <q_black> <q_white> [naive|rgb]\n";
        return 1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    std::string output_image = argv[2];
    std::string output_collage_image = argv[3];
    const double q_black = std::stod(argv[4]);
    const double q_white = std::stod(argv[5]);
    const std::string method = argv[6];
    if (img.empty()) {
        std::cerr << "Could not read the image: " << argv[1] << std::endl;
        return 1;
    }

    cv::Mat contrasted;

    if (method == "naive")
    {
        contrasted = semcv::autocontrast(img, q_black, q_white);
    } else if (method == "rgb")
    {
        contrasted = semcv::autocontrast_rgb(img, q_black, q_white);
    }


    auto create_histogram = [](const cv::Mat& src) {
        cv::Mat gray;
        if (src.channels() == 3) {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = src;
        }

        cv::Mat hist;
        constexpr int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        calcHist(&gray, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);
        normalize(hist, hist, 0, 250, cv::NORM_MINMAX);

        constexpr int pad = 0;  // Отступ слева и справа
        constexpr int width = histSize + 2 * pad;
        constexpr int height = 256;

        cv::Mat histImg(height, width, CV_8UC3, cv::Scalar(240, 240, 240));

        for (int i = 0; i < histSize; i++) {
            const int binVal = cvRound(hist.at<float>(i));
            cv::line(histImg,
                     cv::Point(i + pad, height),
                     cv::Point(i + pad, height - binVal),
                     cv::Scalar(0, 0, 0),
                     1);
        }

        cv::Mat resizedHist;
        resize(histImg, resizedHist, src.size());
        return resizedHist;
    };

    // cv::Mat gray_bgr, contrasted_bgr;
    // cvtColor(gray, gray_bgr, cv::COLOR_GRAY2BGR);
    // cvtColor(contrasted, contrasted_bgr, cv::COLOR_GRAY2BGR);

    cv::Mat hist_orig = create_histogram(img);
    cv::Mat hist_contrasted = create_histogram(contrasted);

    int divider_width = 15;
    cv::Mat divider(img.rows, divider_width, img.type(), cv::Scalar(188, 71, 83));

    cv::Mat top_row, bottom_row, final_image;
    hconcat(std::vector{img, divider,contrasted}, top_row);
    hconcat(std::vector{hist_orig, divider, hist_contrasted}, bottom_row);
    vconcat(std::vector{top_row, bottom_row}, final_image);

    cv::Mat bordered;
    int border_thickness = 10;
    cv::Scalar border_color(188, 71, 83);  // Чёрная рамка

    cv::copyMakeBorder(final_image, bordered,
                       border_thickness, border_thickness,  // top, bottom
                       border_thickness, border_thickness,  // left, right
                       cv::BORDER_CONSTANT, border_color);


    if (!imwrite(output_collage_image, bordered)) {
        std::cerr << "Failed to save output image!" << std::endl;
        return 1;
    }

    if (!imwrite(output_image, contrasted)) {
        std::cerr << "Failed to save output image!" << std::endl;
        return 1;
    }

    std::cout << "Success! Result saved to: " << output_image << std::endl;
    return 0;
}