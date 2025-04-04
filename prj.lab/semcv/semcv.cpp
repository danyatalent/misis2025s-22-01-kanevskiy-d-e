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

