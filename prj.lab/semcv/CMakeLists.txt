add_library(semcv semcv.cpp include/semcv/semcv.hpp)

target_link_libraries(semcv ${OpenCV_LIBS})

set_property(TARGET semcv PROPERTY CXX_STANDARD 20)

target_include_directories(semcv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
