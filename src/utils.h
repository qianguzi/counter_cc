#ifndef TF_DETECTOR_EXAMPLE_UTILS_H
#define TF_DETECTOR_EXAMPLE_UTILS_H

#endif //TF_DETECTOR_EXAMPLE_UTILS_H

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/core/mat.hpp>


using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;


bool isElementInVector(std::vector<int> &v, const int &element);

Status loadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session);

Status readTensorFromMat(const cv::Mat &mat, Tensor &outTensor);

Tensor houghCirclesDetection(const cv::Mat &image, float customRadius=28, float minValue=0, float imageWidth=600, float imageHeight=800);

std::vector<int> getAbnormalIdx(const tensorflow::TTypes<float, 2>::Tensor &boxes, 
                                const tensorflow::TTypes<int, 2>::Tensor &indices);

void drawBoundingBoxOnImage(cv::Mat &image, double xMin, double yMin, double xMax, double yMax, double score, bool scaled);

void drawBoundingBoxesOnImage(cv::Mat &image,
                              const tensorflow::TTypes<float>::Flat &scores,
                              const tensorflow::TTypes<float,2>::Tensor &boxes,
                              std::vector<int> &abnormalIdx);