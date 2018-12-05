#pragma once

#include <string>
#include <set>
#include <opencv2/core/core.hpp>

class Caffe_DetectorPrivate;

struct DetectionResult
{
	cv::Rect rect;
	float score;
	int label;
	std::string name;
};

class ImageDetector
{
private:
	Caffe_DetectorPrivate* detector;
	std::vector<std::string> labels;

public:
	ImageDetector();
	~ImageDetector();
	bool LoadLabels(const std::string& label_file);
	bool LoadDetector(const std::string& model_file, const std::string& weight_file, int size_narrow, int size_long);
	int detect(cv::Mat& image, std::vector<DetectionResult>& results, float conf_thresh);
};
