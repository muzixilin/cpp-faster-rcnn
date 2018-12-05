#include <iostream>
#include <string>
#include <vector>

#include "ImageDetector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void load_detector(ImageDetector& detector, const std::string& model_file,
	const std::string& weight_file, const std::string& label_file, int length_short, int length_long)
{
	
	detector.LoadDetector(model_file, weight_file, length_short, length_long);
	detector.LoadLabels(label_file);
}

void test_image(ImageDetector& detector, std::string& image_file)
{
	cv::Mat image = cv::imread(image_file);

	std::vector<DetectionResult> results;
	detector.detect(image, results, 0.5);
	/*
	for (size_t k = 0; k < results.size(); ++k)
	{
		char textBuf[256];
		sprintf(textBuf, "%s:%g", results[k].name.c_str(), results[k].score);
		cv::putText(image, std::string(textBuf), cv::Point(results[k].rect.x, results[k].rect.y),
			CV_FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2); //标记 类别：置信度 
		cv::rectangle(image, results[k].rect, cv::Scalar(0, 0, 255), 2);
	}

	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", image);
	cv::waitKey(0);
	*/
	std::cout << results.size() << std::endl;
}


int main()
{
#if 1
	ImageDetector detector;
	std::string base_dir("/home/liyu/models/cpp-faster-rcnn/cpp_faster_rcnn/");
	std::string model_file = base_dir + "models/faster_rcnn_test.pt";
	std::string weight_file = base_dir + "models/VGG16_faster_rcnn_final.caffemodel";
	std::string label_file = base_dir + "models/classes.txt";

	load_detector(detector, model_file, weight_file, label_file, 800, 1200);
	
	std::string image_file = base_dir + "images/000001.jpg";
	test_image(detector, image_file);


#else

	std::string image_folder("E:\\data\\fishing_real\\evaluation\\images7x7");
	std::string result_folder("E:\\data\\fishing_real\\evaluation\\detections_frcnn800_7x7");
	evaluate_folder(image_folder, result_folder);

#endif

	std::cout << "finished...\n";
	std::getchar();
}
