#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>

#include <time.h>

#include "utils.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    // Set dirs variables
    string ROOTDIR = "../";
    string GRAPH = "checkpoints/frozen_graph.pb";

    // Set input & output nodes names
    string inputImage = "input_imgs:0";
    string inputLocs = "input_locs:0";
    vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0"};

    // Load and initialize the model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    string imagePath = "./5.jpg";
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;

    Mat oriImage, preImage;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    int FRAME_HEIGHT = 256;
    int FRAME_WIDTH = 256;

    vector<tensorflow::TensorShape> shapes = {tensorflow::TensorShape(), tensorflow::TensorShape()};
    shapes[0].AddDim(1);
    shapes[0].AddDim(FRAME_HEIGHT);
    shapes[0].AddDim(FRAME_WIDTH);
    shapes[0].AddDim(3);
    shapes[1].AddDim(40);
    shapes[1].AddDim(4);

    oriImage = imread(imagePath, 0);
    oriImage.convertTo(preImage, CV_32FC3);
    cvtColor(preImage, preImage, COLOR_GRAY2RGB);
    preImage = preImage / 127.5 - 1;

    // Convert mat to tensor
    inputs[0] = Tensor(tensorflow::DT_FLOAT, shapes[0]);
    inputs[1] = Tensor(tensorflow::DT_FLOAT, shapes[1]);
    Status readTensorStatus = readTensorFromMat(oriImage, inputs[0]);
    if (!readTensorStatus.ok()) {
        LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
        return -1;
    }

    // Run the graph on tensor
    outputs.clear();
    Status runStatus = session->Run({{inputImage, inputs[0]}, {inputLocs, inputs[1]}}, outputLayer, {}, &outputs);
    if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return -1;
    }

    // Extract results from the outputs vector
    tensorflow::TTypes<float, 2>::Tensor resultBoxes = outputs[0].tensor<float, 2>();
    tensorflow::TTypes<float>::Flat resultScores = outputs[1].flat<float>();

    return 0;
}