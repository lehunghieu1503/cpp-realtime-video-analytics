#include "pre-process.hpp"

#include <opencv2/dnn.hpp>

cv::Mat preprocess(const cv::Mat& frame, int targetW, int targetH) {
  // Resize to the model input (stretch, no crop), BGR→RGB, scale to [0, 1]
  // and write straight into the NCHW blob
  return cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(targetW, targetH),
                                cv::Scalar(), /*swapRB=*/true, /*crop=*/false,
                                CV_32F);
}
