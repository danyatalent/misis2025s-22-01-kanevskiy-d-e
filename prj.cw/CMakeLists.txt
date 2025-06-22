add_executable(main_cw main.cpp water_filling.cpp water_filling.h)

target_link_libraries(main_cw ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_include_directories(main_cw PRIVATE ${OpenCV_INCLUDE_DIRS})

add_subdirectory(metric)

install(TARGETS main_cw DESTINATION .)