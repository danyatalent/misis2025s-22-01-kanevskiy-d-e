add_executable(task01_01 task01_01.cpp)
add_executable(task01_02 task01_02.cpp)

target_link_libraries(task01_01 semcv ${OpenCV_LIBS})
target_link_libraries(task01_02 semcv ${OpenCV_LIBS})
