add_executable(create_images_script generate_images.cpp)

target_link_libraries(create_images_script ${OpenCV_LIBS})

