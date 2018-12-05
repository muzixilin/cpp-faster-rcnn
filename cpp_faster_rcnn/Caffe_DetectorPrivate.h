#ifndef DETECTORPRIVATE_H
#define DETECTORPRIVATE_H

#define INPUT_SIZE_NARROW  600
#define INPUT_SIZE_LONG  1000

// #define CPU_ONLY 1
#define USE_OPENCV 1

/*
#ifdef CPU_ONLY
#pragma comment(lib, "libcaffe_cpu.lib")
#else
#pragma comment(lib, "libcaffe_gpu.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cudnn.lib")
#endif
*/
#include <string>
//#include "caffe/caffe.hpp"
#include <caffe/net.hpp>
#include <opencv2/core/core.hpp>
#include <memory>

struct Result
{
	float arry[6];
};

class Caffe_DetectorPrivate
{
public:
	Caffe_DetectorPrivate(int size_narrow = 600, int size_long = 1000);
	bool loadModel(const std::string &model_file, const std::string &weights_file,
		const std::string &mean_file, const std::string &label_file, bool encrypt_flag = 0);

	/*DetectResult*/
	void detect(cv::Mat &image, const float fThrsh = 0.3f);

	void setInputSize(int size_narrow, int size_long);
private:
	std::shared_ptr< caffe::Net<float> > net_;
	int class_num_;
	
	int input_size_narrow;
	int input_size_long;

public:
	std::vector<Result> m_Result;
};

namespace RPN{
	struct abox
	{
		float x1;
		float y1;
		float x2;
		float y2;
		float score;
		float label;
		bool operator <(const abox&tmp) const{
			return score < tmp.score;
		}
	};
	void nms(std::vector<abox>& input_boxes, float nms_thresh);
	cv::Mat bbox_tranform_inv(cv::Mat, cv::Mat);
}

#endif
