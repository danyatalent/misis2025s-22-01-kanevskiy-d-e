#include <semcv/semcv.hpp>

namespace semcv
{
    std::string mat_type_to_str(const int type) {
        switch (type) {
        case CV_8U:  return "uint08";
        case CV_8S:  return "sint08";
        case CV_16U: return "uint16";
        case CV_16S: return "sint16";
        case CV_32S: return "sint32";
        case CV_32F: return "real32";
        case CV_64F: return "real64";
        default:     return "unknown";
        }
    }

    std::string strid_from_mat(const cv::Mat& img, const int n) {
        std::ostringstream ss;
        ss << std::setw(n) << std::setfill('0') << img.cols << "x"
           << std::setw(n) << std::setfill('0') << img.rows << "."
           << img.channels() << "."
           << mat_type_to_str(img.depth());
        return ss.str();
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

    cv::Mat generate_gray_stripes_mat()
    {
        cv::Mat img(30, 768, CV_8UC1);
        cv::Mat row(1, img.cols, CV_8UC1);
        for (int x = 0; x < img.cols; ++x) {
            row.at<uchar>(0, x) = static_cast<uchar>(x / 3);
        }

        for (int y = 0; y < img.rows; ++y) {
            row.copyTo(img.row(y));
        }

        return img;
    }

    cv::Mat gamma_correction(const cv::Mat& img, const double gamma) {
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        cv::Mat res;
        cv::LUT(img, lookUpTable, res);
        return res;
    }

    cv::Mat gen_tgtimg00(const int lev0, const int lev1, const int lev2)
    {
        constexpr int size = 256;
        constexpr int square_side = 209;
        constexpr int circle_radius = 83;

        cv::Mat img(size, size, CV_8UC1, cv::Scalar(lev0));

        const cv::Point center(size / 2, size / 2);

        constexpr int half_square = square_side / 2;
        const cv::Point top_left(center.x - half_square, center.y - half_square);
        const cv::Point bottom_right(center.x + half_square, center.y + half_square);

        cv::rectangle(img, top_left, bottom_right, cv::Scalar(lev1), cv::FILLED);

        cv::circle(img, center, circle_radius, cv::Scalar(lev2), cv::FILLED);

        return img;
    }

    cv::Mat add_noise_gau(const cv::Mat& img, const int std)
    {
        CV_Assert(img.type() == CV_8UC1);

        cv::Mat img_f;
        img.convertTo(img_f, CV_32F);

        cv::Mat noise(img.size(), CV_32F);
        cv::randn(noise, 0.0f, static_cast<float>(std));

        const cv::Mat noisy_f = img_f + noise;

        cv::Mat noisy_clamped;
        cv::threshold(noisy_f, noisy_clamped, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(noisy_clamped, noisy_clamped, 0, 0, cv::THRESH_TOZERO);

        cv::Mat noisy_u8;
        noisy_clamped.convertTo(noisy_u8, CV_8U);

        return noisy_u8;
    }

    DistributionStats compute_stats(const cv::Mat& img, const cv::Mat& mask) {
        CV_Assert(img.type() == CV_8UC1 && mask.type() == CV_8UC1);
        cv::Scalar mean, stddev;
        cv::meanStdDev(img, mean, stddev, mask);
        return { mean[0], stddev[0] };
    }

    void create_masks(const int size, const int square_side, const int circle_radius,
                  cv::Mat& mask_bg, cv::Mat& mask_square, cv::Mat& mask_circle) {
        const cv::Point center(size / 2, size / 2);
        mask_bg = cv::Mat(size, size, CV_8UC1, cv::Scalar(255));
        mask_square = cv::Mat::zeros(size, size, CV_8UC1);
        mask_circle = cv::Mat::zeros(size, size, CV_8UC1);

        const int half = square_side / 2;
        cv::rectangle(mask_square,
                      cv::Point(center.x - half, center.y - half),
                      cv::Point(center.x + half, center.y + half),
                      cv::Scalar(255), cv::FILLED);

        cv::circle(mask_circle, center, circle_radius, cv::Scalar(255), cv::FILLED);

        mask_bg.setTo(0, mask_square);
        mask_square.setTo(0, mask_circle);
    }

    cv::Mat draw_histogram(const cv::Mat& img_input, const cv::Scalar& bg_color) {
        CV_Assert(img_input.type() == CV_8UC1);

        constexpr int width = 256;
        constexpr int height = 256;
        cv::Mat hist_img(height, width, CV_8UC1, bg_color);

        constexpr int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::Mat hist;

        cv::calcHist(&img_input, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);
        cv::normalize(hist, hist, 0, 250, cv::NORM_MINMAX);

        for (int i = 0; i < 256; ++i) {
            const int val = cvRound(hist.at<float>(i));
            cv::line(hist_img, cv::Point(i, height), cv::Point(i, height - val),
                     cv::Scalar(30), 1, cv::LINE_8);
        }

        return hist_img;
    }

    cv::Mat make_histogram_grid(const std::vector<cv::Mat>& images) {
        std::vector<cv::Mat> rows;
        bool alt = false;

        const size_t num_rows = images.size() / 4;
        for (size_t i = 0; i < num_rows; ++i) {
            std::vector<cv::Mat> row;
            for (size_t j = 0; j < 4; ++j) {
                const size_t index = i * 4 + j;
                if (index >= images.size()) {
                    break;
                }
                cv::Scalar bg = (alt ^ (j % 2 == 0)) ? cv::Scalar(230) : cv::Scalar(180);
                row.push_back(draw_histogram(images[index], bg));
            }

            while (row.size() < 4) {
                row.push_back(cv::Mat::zeros(256, 256, CV_8UC1));
            }
            cv::Mat concat_row;
            cv::hconcat(row, concat_row);
            rows.push_back(concat_row);
            alt = !alt;
        }

        cv::Mat result;
        if (!rows.empty()) {
            cv::vconcat(rows, result);
        } else {
            result = cv::Mat::zeros(256, 256, CV_8UC1);
        }
        return result;
    }

    cv::Mat autocontrast(const cv::Mat& img, const double q_black, const double q_white) {
        CV_Assert(!img.empty());
        CV_Assert(q_black >= 0.0 && q_black <= 1.0);
        CV_Assert(q_white >= 0.0 && q_white <= 1.0);
        CV_Assert(q_black <= q_white);

        if (img.channels() == 3)
        {
            return naive_autocontrast(img, q_black, q_white);
        }

        const int total_pixels = img.rows * img.cols;

        int hist[256] = {0};
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                hist[img.at<uchar>(y, x)]++;
            }
        }

        double cumsum[256] = {0};
        cumsum[0] = hist[0];
        for (int i = 1; i < 256; i++) {
            cumsum[i] = cumsum[i-1] + hist[i];
        }

        const double black_threshold = q_black * total_pixels;
        const double white_threshold = q_white * total_pixels;

        uchar v_min = 0;
        uchar v_max = 255;

        for (int i = 0; i < 256; i++) {
            if (cumsum[i] >= black_threshold) {
                v_min = i;
                break;
            }
        }

        for (int i = 255; i >= 0; i--) {
            if (cumsum[i] <= white_threshold) {
                v_max = i;
                break;
            }
        }

        if (v_min == v_max) {
            return img.clone();
        }

        cv::Mat result;
        img.convertTo(result, CV_8UC3, 255.0 / (v_max - v_min), -255.0 * v_min / (v_max - v_min));

        cv::threshold(result, result, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(result, result, 0, 0, cv::THRESH_TOZERO);

        return result;
    }

    cv::Mat naive_autocontrast(const cv::Mat& img, const double q_black, const double q_white)
    {
        CV_Assert(!img.empty());
        CV_Assert(img.type() == CV_8UC3);
        CV_Assert(q_black >= 0.0 && q_black <= 1.0);
        CV_Assert(q_white >= 0.0 && q_white <= 1.0);

        std::vector<cv::Mat> channels;
        cv::split(img, channels);

        for (auto& channel : channels)
        {
            channel = autocontrast(channel, q_black, q_white);
        }

        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }


    cv::Mat autocontrast_rgb(const cv::Mat& img, const double q_black, const double q_white) {
        CV_Assert(img.type() == CV_8UC3);
        CV_Assert(0.0 <= q_black && q_black < q_white && q_white <= 1.0);

        std::vector<cv::Mat> channels;
        cv::split(img, channels);

        // Собираем все значения пикселей каждого канала в один вектор
        std::vector<uchar> all_pixels_r, all_pixels_g, all_pixels_b;
        all_pixels_r.assign(channels[2].begin<uchar>(), channels[2].end<uchar>());
        all_pixels_g.assign(channels[1].begin<uchar>(), channels[1].end<uchar>());
        all_pixels_b.assign(channels[0].begin<uchar>(), channels[0].end<uchar>());

        // Получаем объединённый вектор интенсивностей (для согласованности)
        std::vector<uchar> all_pixels;
        all_pixels.reserve(all_pixels_r.size() * 3);
        all_pixels.insert(all_pixels.end(), all_pixels_r.begin(), all_pixels_r.end());
        all_pixels.insert(all_pixels.end(), all_pixels_g.begin(), all_pixels_g.end());
        all_pixels.insert(all_pixels.end(), all_pixels_b.begin(), all_pixels_b.end());

        std::ranges::sort(all_pixels);

        auto get_quantile = [&](double q) {
            const auto index = static_cast<size_t>(q * (all_pixels.size() - 1));
            return all_pixels[index];
        };

        const uchar low = get_quantile(q_black);
        const uchar high = get_quantile(q_white);
        if (low >= high) return img.clone();  // avoid division by zero

        // Линейное растяжение
        cv::Mat result = img.clone();
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const uchar v = img.at<cv::Vec3b>(y, x)[c];
                    uchar stretched;
                    if (v <= low)
                        stretched = 0;
                    else if (v >= high)
                        stretched = 255;
                    else
                        stretched = static_cast<uchar>((v - low) * 255.0 / (high - low));
                    result.at<cv::Vec3b>(y, x)[c] = stretched;
                }
            }
        }

        return result;
    }

}
