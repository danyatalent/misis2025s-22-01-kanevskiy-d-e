add_executable(task04_01 task04_01.cpp)

target_link_libraries(task04_01 nlohmann_json::nlohmann_json ${OpenCV_LIBS})

add_executable(task04_02 task04_02.cpp)

target_link_libraries(task04_02 nlohmann_json::nlohmann_json ${OpenCV_LIBS})

add_executable(task04_03 task04_03.cpp)

target_link_libraries(task04_03 nlohmann_json::nlohmann_json ${OpenCV_LIBS})