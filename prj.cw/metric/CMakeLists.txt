add_executable(calculate_metric metric.cpp)

target_link_libraries(calculate_metric ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_include_directories(calculate_metric PRIVATE ${OpenCV_INCLUDE_DIRS})

install(TARGETS calculate_metric DESTINATION .)