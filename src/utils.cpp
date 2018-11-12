#include "utils.h"

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv/cv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


bool isElementInVector(vector<int> &v, const int &element) {
	vector<int>::iterator it;
	it=find(v.begin(),v.end(),element);
	if (it!=v.end()) {
		return true;
	}
	else {
		return false;
	}
}
/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status loadGraph(const string &graph_file_name,
                 unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
 */
Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {

    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.copyTo(fakeMat);

    return Status::OK();
}


Tensor houghCirclesDetection(const Mat &image, float customWidth, float customHeight, float minValue, float imageWidth, float imageHeight) {
    Tensor houghBoxesTensor;
    tensorflow::TensorShape boxShape = tensorflow::TensorShape();
    std::vector<Vec3f> circles;
    Mat preImage;

    equalizeHist(image, preImage);
    GaussianBlur(preImage, preImage, {5, 5}, 0);
    Laplacian(preImage, preImage, -1, 5);
    medianBlur(preImage, preImage, 5);
    HoughCircles(preImage, circles, HOUGH_GRADIENT, 1, 55, 100, 35, 10, 28);

    int circleSize = int(circles.size());
    boxShape.AddDim(circleSize);
    boxShape.AddDim(4);
    houghBoxesTensor = Tensor(tensorflow::DT_FLOAT, boxShape);
    tensorflow::TTypes<float, 2>::Tensor houghBoxes = houghBoxesTensor.tensor<float, 2>();
    //vector<vector<float>> houghBoxes;
    for (int i=0; i<circleSize; i++)
    {
        houghBoxes(i, 0) = max(minValue, circles[i][0]-customWidth)/imageWidth;
        houghBoxes(i, 1) = max(minValue, circles[i][1]-customHeight)/imageHeight;
        houghBoxes(i, 2) = min(imageWidth, circles[i][0]+customWidth)/imageWidth;
        houghBoxes(i, 3) = min(imageHeight, circles[i][1]+customHeight)/imageHeight;
    }
    return houghBoxesTensor;
}

std::vector<int> getAbnormalIdx(const tensorflow::TTypes<float, 2>::Tensor &boxes, 
                                const tensorflow::TTypes<float>::Flat &scores,
                                const tensorflow::TTypes<float, 2>::Tensor &inter,
                                const tensorflow::TTypes<int>::Flat &interIdx,
                                const tensorflow::TTypes<int, 2>::Tensor &indices) {
    vector<int> abnormalIdx;
    float centerL, centerU, centerR, centerD;
    float idxL = indices(0,0);
    float idxU = indices(0,1);
    float idxR = indices(3,0);
    float idxD = indices(3,1);
    int oriNumBoxes = scores.size();

    if (interIdx.size() > 0){
        for (int i = 0; i < interIdx.size(); i++){
            for (int j = 0; j < oriNumBoxes; j++){
                if ((inter(interIdx(i), j) == 1) && ((scores(j) - scores(interIdx(i))) > 0.1)){
                    abnormalIdx.push_back(interIdx(i));
                    break;
                }
            }
        }
    }
    centerL = (boxes(idxL, 2)-boxes(idxL, 0))/2 + boxes(idxL, 0);
    centerU = (boxes(idxU, 3)-boxes(idxU, 1))/2 + boxes(idxU, 1);
    centerR = (boxes(idxR, 2)-boxes(idxR, 0))/2 + boxes(idxR, 0);
    centerD = (boxes(idxD, 3)-boxes(idxD, 1))/2 + boxes(idxD, 1);

    if (centerL<boxes(indices(1,0), 0)) {
        if (not isElementInVector(abnormalIdx, idxL))
            abnormalIdx.push_back(idxL);
    }
    if (centerU<boxes(indices(1,1), 1)) {
        if (not isElementInVector(abnormalIdx, idxU))
            abnormalIdx.push_back(idxU);
    }
    if (centerR>boxes(indices(2,0), 2)) {
        if (not isElementInVector(abnormalIdx, idxR))
            abnormalIdx.push_back(idxR);
    }
    if (centerD>boxes(indices(2,1), 3)) {
        if (not isElementInVector(abnormalIdx, idxD))
            abnormalIdx.push_back(idxD);
    }
    return abnormalIdx;
}
/** Draw bounding box and add scoreString to the image.
 *  Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
 */
void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, bool scaled=true) {
    cv::Point tl, br;
    if (scaled) {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    } else {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }
    cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

    // Ceiling the score down to 3 decimals (weird!)
    float scoreRounded = floorf(score * 1000) / 1000;
    string scoreString = to_string(scoreRounded).substr(0, 5);

    int fontCoeff = 12;
    cv::Point brRect = cv::Point(tl.x + scoreString.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, scoreString, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

/** Draw bounding boxes and add captions to the image.
 *  Box is drawn only if corresponding score is higher than the _threshold_.
 */
void drawBoundingBoxesOnImage(Mat &image,
                              const tensorflow::TTypes<float>::Flat &scores,
                              const tensorflow::TTypes<float,2>::Tensor &boxes,
                              std::vector<int> &abnormalIdx) {
    for (int j = 0; j < scores.size(); j++)
        if (not isElementInVector(abnormalIdx, j))
            drawBoundingBoxOnImage(image,
                                boxes(j,1), boxes(j,0),
                                boxes(j,3), boxes(j,2),
                                scores(j));
}