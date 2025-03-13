#include <semcv/semcv.hpp>

std::string mat_type_to_str(int type) {
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

void generateAndSaveImages(const std::string& folder) {
    std::vector<int> depths = {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F};
    std::vector<int> channels = {1, 3};
    std::vector<std::string> formats = {"png", "tiff", "jpg"};
    
    for (int depth : depths) {
        for (int ch : channels) {
            cv::Mat img(100, 100, CV_MAKETYPE(depth, ch), cv::Scalar::all(128));
            std::string id = strid_from_mat(img);
            for (const std::string& format : formats) {
                std::string filename = folder + "/" + id + "." + format;
                cv::imwrite(filename, img);
            }
        }
    }
}
