add_executable(psnr psnr.cpp)

target_link_libraries(psnr ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_include_directories(psnr PRIVATE ${OpenCV_INCLUDE_DIRS})