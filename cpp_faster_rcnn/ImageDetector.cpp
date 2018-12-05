#include "ImageDetector.h"
#include "Caffe_DetectorPrivate.h"
#include <fstream>

ImageDetector::ImageDetector()
{
	detector = NULL;
}

ImageDetector::~ImageDetector()
{
	if (detector != NULL)
		delete detector;
}

bool ImageDetector::LoadLabels(const std::string& label_file)
{
	// read labels from file
	labels.clear();
	std::ifstream infile(label_file.c_str());
	if (!infile.is_open())
		return false;
	std::string str;
	while (std::getline(infile, str))
	{
		if (str.size() == 0)
			continue;
		labels.push_back(str);
	}

#if 0
	// output labels
	for (size_t i = 0; i < labels.size(); ++i)
	{
		OutputDebugStringA(labels[i].c_str());
		OutputDebugStringA("---\n");
	}
#endif

	return true;
}

bool ImageDetector::LoadDetector(const std::string& model_file, const std::string& weight_file,
	int size_narrow, int size_long)
{
	if (detector == NULL)
		detector = new Caffe_DetectorPrivate(size_narrow, size_long);
	detector->loadModel(model_file, weight_file, "", "");
	return true;
}

int ImageDetector::detect(cv::Mat& image,  std::vector<DetectionResult>& results, float conf_thresh)
{
	if (detector == NULL)
		return -1;

	detector->detect(image);
	results.clear();

	int count = 0;
	for (size_t i = 0; i < detector->m_Result.size(); ++i)
	{
		Result d = detector->m_Result[i];
		float score = d.arry[1];
		if (score > conf_thresh)
		{
			count += 1;
			DetectionResult res;

			cv::Point pt1, pt2;
			pt1.x = static_cast<int>(d.arry[2]);
			pt1.y = static_cast<int>(d.arry[3]);
			pt2.x = static_cast<int>(d.arry[4]);
			pt2.y = static_cast<int>(d.arry[5]);

			res.rect = cv::Rect(pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
			res.score = score;
			res.label = static_cast<int>(d.arry[0]);
			if (res.label >= 0 && res.label < int(labels.size()))
				res.name = labels[res.label];
			else
				res.name = "unknown";

			results.push_back(res);
		}
	}
	return count;
}
