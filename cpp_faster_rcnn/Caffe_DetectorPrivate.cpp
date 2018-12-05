#include "Caffe_DetectorPrivate.h"
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <cmath>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <ostream>
#include <iostream>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::vector;
using std::max;
using std::min;
#endif
//Using for box sort

namespace RPN{
	cv::Mat bbox_tranform_inv(cv::Mat local_anchors, cv::Mat boxs_delta){
		cv::Mat pre_box(local_anchors.rows, local_anchors.cols, CV_32FC1);
		for (int i = 0; i < local_anchors.rows; i++)
		{
			double pred_ctr_x, pred_ctr_y, src_ctr_x, src_ctr_y;
			double dst_ctr_x, dst_ctr_y, dst_scl_x, dst_scl_y;
			double src_w, src_h, pred_w, pred_h;
			src_w = local_anchors.at<float>(i, 2) - local_anchors.at<float>(i, 0) + 1;
			src_h = local_anchors.at<float>(i, 3) - local_anchors.at<float>(i, 1) + 1;
			src_ctr_x = local_anchors.at<float>(i, 0) + 0.5 * src_w;
			src_ctr_y = local_anchors.at<float>(i, 1) + 0.5 * src_h;

			dst_ctr_x = boxs_delta.at<float>(i, 0);
			dst_ctr_y = boxs_delta.at<float>(i, 1);
			dst_scl_x = boxs_delta.at<float>(i, 2);
			dst_scl_y = boxs_delta.at<float>(i, 3);
			pred_ctr_x = dst_ctr_x*src_w + src_ctr_x;
			pred_ctr_y = dst_ctr_y*src_h + src_ctr_y;
			pred_w = exp(dst_scl_x) * src_w;
			pred_h = exp(dst_scl_y) * src_h;

			pre_box.at<float>(i, 0) = pred_ctr_x - 0.5*pred_w;
			pre_box.at<float>(i, 1) = pred_ctr_y - 0.5*pred_h;
			pre_box.at<float>(i, 2) = pred_ctr_x + 0.5*pred_w;
			pre_box.at<float>(i, 3) = pred_ctr_y + 0.5*pred_h;
		}
		return pre_box;
	}


	void nms(std::vector<abox> &input_boxes, float nms_thresh){
		std::vector<float>vArea(input_boxes.size());
		for (size_t i = 0; i < input_boxes.size(); ++i)
		{
			vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
				* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
		}
		for (size_t i = 0; i < input_boxes.size(); ++i)
		{
			for (size_t j = i + 1; j < input_boxes.size();)
			{
				float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
				float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
				float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
				float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
				float w = std::max(float(0), xx2 - xx1 + 1);
				float   h = std::max(float(0), yy2 - yy1 + 1);
				float   inter = w * h;
				float ovr = inter / (vArea[i] + vArea[j] - inter);
				if (ovr > nms_thresh)
				{
					input_boxes.erase(input_boxes.begin() + j);
					vArea.erase(vArea.begin() + j);
				}
				else
				{
					j++;
				}
			}
		}
	}
}

void RestoreBlobFromtxt(char* szSaveFileName, Blob<float>& boxes, Blob<float>& scores)
{
	if (!szSaveFileName){
		return;
	}
	std::ifstream infile;
	infile.open(szSaveFileName, std::ios::in);
	if (!infile.is_open()) return;

	int n = scores.shape(0);
	int c = scores.shape(1);
	int h = scores.shape(2);
	int w = scores.shape(3);
	// int channel_size = w*h;
	float tt = 0.0f;
	for (int i = 0; i < n; i++)
	{
		for (int p = 0; p < h; p++)
		{
			for (int q = 0; q < w; q++)
			{
				for (int j = 0; j < c / 2; j++)
				{
					for (int l = 0; l < 4; l++)
					{
						//outfile << boxes.data_at(i, 4 * j + l, p, q) << " ";
						infile >> tt;
					}
					//outfile << scores.data_at(i, j + c / 2, p, q) << std::endl;
					infile >> tt;
				}
			}

		}

	}
	infile.close();
}

Caffe_DetectorPrivate::Caffe_DetectorPrivate(int size_narrow, int size_long)
{
	input_size_narrow = size_narrow;
	input_size_long = size_long;
}

void Caffe_DetectorPrivate::setInputSize(int size_narrow, int size_long)
{
	input_size_narrow = size_narrow;
	input_size_long = size_long; 
}

bool Caffe_DetectorPrivate::loadModel(const std::string &model_file, const std::string &weights_file,
	const std::string &mean_file, const std::string &label_file, bool encrypt_flag)
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif //#ifdef CPU_ONLY 
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(weights_file);
	class_num_ = net_->blob_by_name("cls_prob")->channels();
	return true;
}


void Caffe_DetectorPrivate::detect(cv::Mat &image, const float fThrsh)
{
	float CONF_THRESH = fThrsh;
	float NMS_THRESH = 0.3;

	int max_side = max(image.rows, image.cols);
	int min_side = min(image.rows, image.cols);

	float max_side_scale = float(max_side) / float(input_size_long);
	float min_side_scale = float(min_side) / float(input_size_narrow);
	float max_scale = max(max_side_scale, min_side_scale);

	float img_scale = float(1) / max_scale;
	int height = int(image.rows * img_scale);
	int width = int(image.cols * img_scale);
	//printf("%d,%d", height, width);
	// int num_out;
	cv::Mat cv_resized;
	image.convertTo(cv_resized, CV_32FC3);
	cv::Mat normalized;
	cv::Mat mean(image.rows, image.cols, cv_resized.type(), cv::Scalar(102.9801, 115.9465, 122.7717));
	subtract(cv_resized, mean, cv_resized);
	cv::resize(cv_resized, normalized, cv::Size(width, height));

	float im_info[3];
	im_info[0] = height;
	im_info[1] = width;
	im_info[2] = img_scale;
	shared_ptr<Blob<float>> input_layer = net_->blob_by_name("data");
	input_layer->Reshape(1, normalized.channels(), height, width);
	net_->Reshape();
	float* input_data = input_layer->mutable_cpu_data();
	vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += height * width;
	}
	cv::split(normalized, input_channels);
	net_->blob_by_name("im_info")->set_cpu_data(im_info);
	// net_->Forward();
	net_->ForwardPrefilled();


	int num = net_->blob_by_name("rois")->num();
	const float *rois_data = net_->blob_by_name("rois")->cpu_data();
	// int num1 = net_->blob_by_name("bbox_pred")->num();
	cv::Mat rois_box(num, 4, CV_32FC1);
	for (int i = 0; i < num; ++i)
	{
		rois_box.at<float>(i, 0) = rois_data[i * 5 + 1] / img_scale;
		rois_box.at<float>(i, 1) = rois_data[i * 5 + 2] / img_scale;
		rois_box.at<float>(i, 2) = rois_data[i * 5 + 3] / img_scale;
		rois_box.at<float>(i, 3) = rois_data[i * 5 + 4] / img_scale;
	}

	shared_ptr<Blob<float>> bbox_delt_data = net_->blob_by_name("bbox_pred");
	shared_ptr<Blob<float>>score = net_->blob_by_name("cls_prob");

	vector<RPN::abox> result;
	for (int i = 1; i < class_num_; ++i)
	{
		cv::Mat bbox_delt(num, 4, CV_32FC1);
		for (int j = 0; j < num; ++j)
		{
			bbox_delt.at<float>(j, 0) = bbox_delt_data->data_at(j, i * 4, 0, 0);
			bbox_delt.at<float>(j, 1) = bbox_delt_data->data_at(j, i * 4 + 1, 0, 0);
			bbox_delt.at<float>(j, 2) = bbox_delt_data->data_at(j, i * 4 + 2, 0, 0);
			bbox_delt.at<float>(j, 3) = bbox_delt_data->data_at(j, i * 4 + 3, 0, 0);
		}
		cv::Mat box_class = RPN::bbox_tranform_inv(rois_box, bbox_delt);

		vector<RPN::abox>aboxes;
		for (int j = 0; j < box_class.rows; ++j)
		{
			if (box_class.at<float>(j, 0) < 0)  box_class.at<float>(j, 0) = 0;
			if (box_class.at<float>(j, 0) > (image.cols - 1))   box_class.at<float>(j, 0) = image.cols - 1;
			if (box_class.at<float>(j, 2) < 0)  box_class.at<float>(j, 2) = 0;
			if (box_class.at<float>(j, 2) > (image.cols - 1))   box_class.at<float>(j, 2) = image.cols - 1;

			if (box_class.at<float>(j, 1) < 0)  box_class.at<float>(j, 1) = 0;
			if (box_class.at<float>(j, 1) > (image.rows - 1))   box_class.at<float>(j, 1) = image.rows - 1;
			if (box_class.at<float>(j, 3) < 0)  box_class.at<float>(j, 3) = 0;
			if (box_class.at<float>(j, 3) > (image.rows - 1))   box_class.at<float>(j, 3) = image.rows - 1;
			RPN::abox tmp;
			tmp.x1 = box_class.at<float>(j, 0);
			tmp.y1 = box_class.at<float>(j, 1);
			tmp.x2 = box_class.at<float>(j, 2);
			tmp.y2 = box_class.at<float>(j, 3);
			tmp.score = score->data_at(j, i, 0, 0);
			tmp.label = i;
			aboxes.push_back(tmp);
		}
		std::sort(aboxes.rbegin(), aboxes.rend());
		RPN::nms(aboxes, NMS_THRESH);
		for (size_t k = 0; k < aboxes.size();)
		{
			if (aboxes[k].score < CONF_THRESH)
			{
				aboxes.erase(aboxes.begin() + k);
			}
			else
			{
				k++;
			}
		}
		result.insert(result.end(), aboxes.begin(), aboxes.end());
	}
	m_Result.clear();
	for (size_t i = 0; i < result.size(); i++)
	{
		Result res;
		res.arry[0] = result[i].label;
		res.arry[1] = result[i].score;
		res.arry[2] = result[i].x1;
		res.arry[3] = result[i].y1;
		res.arry[4] = result[i].x2;
		res.arry[5] = result[i].y2;
		m_Result.push_back(res);
	}
	return;
}
