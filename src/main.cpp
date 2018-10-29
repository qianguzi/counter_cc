#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/ops.h"
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

#include <sys/time.h>

#include "utils.h"

using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // Set input & output nodes names
    string inputImageNode = "input_img:0";
    string inputGridNode = "input_grid_size:0";
    string inputBoxesNode = "input_boxes:0";
    vector<string> outputLayer = {"Non_max_suppression/result_boxes:0", \
                                "Non_max_suppression/result_scores:0", \
                                "Non_max_suppression/abnormal_indices"};

    // Load and initialize the model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    string graphPath = "../checkpoints/counter_v2/model.ckpt-150000.pb";
    string imageFilePath = "../dataset/name.txt";
    string imageDir = "../dataset/img/";
    string resultImageDir = "../dataset/result/";
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok())
    {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    }
    else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;

    struct timeval start, end;
    Mat oriImage, inputImage, colorImage;
    Tensor inputImageTensor, inputGridTensor;
    std::vector<Tensor> outputTensors;
    tensorflow::TensorShape imgShape = tensorflow::TensorShape();
    tensorflow::TensorShape gridSizeShape = tensorflow::TensorShape();

    imgShape.AddDim(1);
    imgShape.AddDim(800);
    imgShape.AddDim(600);
    imgShape.AddDim(3);
    gridSizeShape.AddDim(2);

    inputImageTensor = Tensor(tensorflow::DT_FLOAT, imgShape);
    inputGridTensor = Tensor(tensorflow::DT_FLOAT, gridSizeShape);
    tensorflow::TTypes<float, 1>::Tensor grid_size = inputGridTensor.tensor<float, 1>();
    grid_size.setValues({5.0, 3.0});

    ifstream fin(imageFilePath);
    string s;
    while (getline(fin, s))
    {
        gettimeofday(&start, NULL);
        oriImage = imread(imageDir + s + "_resize.jpg", 0);
        cvtColor(oriImage, colorImage, COLOR_GRAY2RGB);
        colorImage.convertTo(inputImage, CV_32FC3);
        inputImage = inputImage / 127.5 - 1;
        
        Tensor inputBoxesTensor = houghCirclesDetection(oriImage);

        // Convert mat to tensor
        Status readTensorStatus = readTensorFromMat(inputImage, inputImageTensor);
        if (!readTensorStatus.ok())
        {
            LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
            return -1;
        }

        // Run the graph on tensor
        outputTensors.clear();
        Status runStatus = session->Run({{inputImageNode, inputImageTensor}, \
                                        {inputGridNode, inputGridTensor}, \
                                        {inputBoxesNode, inputBoxesTensor}}, \
                                        outputLayer, {}, &outputTensors);
        if (!runStatus.ok())
        {
            LOG(ERROR) << "Running model failed: " << runStatus;
            return -1;
        }

        // Extract results from the outputTensors vector
        tensorflow::TTypes<float, 2>::Tensor resultBoxes = outputTensors[0].tensor<float, 2>();
        tensorflow::TTypes<float>::Flat resultScores = outputTensors[1].flat<float>();
        tensorflow::TTypes<int, 2>::Tensor abnormalIndices = outputTensors[2].tensor<int, 2>();

        vector<int> abnormalIdx = getAbnormalIdx(resultBoxes, abnormalIndices);
        
        gettimeofday(&end, NULL);
        cout << 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec << endl;

        drawBoundingBoxesOnImage(colorImage, resultScores, resultBoxes, abnormalIdx);

        imwrite(resultImageDir + s + "result.jpg", colorImage);
    }
    return 0;
}
