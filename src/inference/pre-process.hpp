#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Preprocess a BGR frame for YOLOv8 ONNX inference
// Steps: resize to target → BGR→RGB → normalize [0,1] → HWC→CHW
// Returns a float32 blob of shape [1, 3, targetH, targetW] ready for the ONNX
// input tensor
cv::Mat preprocess(const cv::Mat& frame, int targetW, int targetH);
