add_executable(task06 task06.cpp)

target_link_libraries(task06 ${OpenCV_LIBS} nlohmann_json::nlohmann_json)