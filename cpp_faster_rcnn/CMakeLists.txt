# cmake script for faster rcnn of cpp version
cmake_minimum_required (VERSION 2.8)

project (faster_rcnn_demo)

# set c++11 
if (UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

# add source files
add_executable(faster_rcnn_demo main.cpp ImageDetector.cpp Caffe_DetectorPrivate.cpp)

# add include paths
include_directories ( /home/liyu/models/cpp-faster-cnn/cpp_faster_rcnn/
    /home/liyu/models/cpp-faster-rcnn/caffe-fast-rcnn/include/
    /usr/local/include/ 
    /usr/local/cuda/include/ )

# add shared library paths
target_link_libraries(faster_rcnn_demo /home/liyu/models/cpp-faster-rcnn/caffe-fast-rcnn/build/lib/libcaffe.so
    /usr/local/lib/libopencv_highgui.so 
    /usr/local/lib/libopencv_core.so 
    /usr/local/lib/libopencv_imgproc.so 
    /usr/local/lib/libopencv_imgcodecs.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
    /usr/lib/x86_64-linux-gnu/libboost_thread.so
    /usr/lib/x86_64-linux-gnu/libboost_date_time.so
    /usr/lib/x86_64-linux-gnu/libboost_chrono.so
    )

