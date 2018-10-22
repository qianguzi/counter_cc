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

/** Draw bounding box and add caption to the image.
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
    string caption = "cap (" + scoreString + ")";

    // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
    int fontCoeff = 12;
    cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

/** Draw bounding boxes and add captions to the image.
 *  Box is drawn only if corresponding score is higher than the _threshold_.
 */
void drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float,2>::Tensor &boxes) {
    for (int j = 0; j < scores.size(); j++)
        drawBoundingBoxOnImage(image,
                               boxes(j,0), boxes(j,1),
                               boxes(j,2), boxes(j,3),
                               scores(j));
}