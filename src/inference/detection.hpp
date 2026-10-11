#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Object detection result from YOLOv8 inference
struct Detection {
  int classId;
  float confidence;
  cv::Rect2f bbox;  // x, y, width, height in pixel coordinates
  int trackId = -1;  // set by the tracker; -1 while the object is untracked
};

// COCO dataset class names (80 classes) — matches YOLOv8 default training
// clang-format off
inline const std::array<const char*, 80> kCocoClassNames = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",
    "bus",            "train",      "truck",         "boat",          "traffic light",
    "fire hydrant",   "stop sign",  "parking meter", "bench",         "bird",
    "cat",            "dog",        "horse",         "sheep",         "cow",
    "elephant",       "bear",       "zebra",         "giraffe",       "backpack",
    "umbrella",       "handbag",    "tie",           "suitcase",      "frisbee",
    "skis",           "snowboard",  "sports ball",   "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",
    "wine glass",     "cup",        "fork",          "knife",         "spoon",
    "bowl",           "banana",     "apple",         "sandwich",      "orange",
    "broccoli",       "carrot",     "hot dog",       "pizza",         "donut",
    "cake",           "chair",      "couch",         "potted plant",  "bed",
    "dining table",   "toilet",     "tv",            "laptop",        "mouse",
    "remote",         "keyboard",   "cell phone",    "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",
    "vase",           "scissors",   "teddy bear",    "hair drier",    "toothbrush"
};
// clang-format on

// Get class name safely — returns "unknown" if classId is out of range
inline const char* getClassName(int classId) {
  if (classId < 0 || classId >= static_cast<int>(kCocoClassNames.size()))
    return "unknown";
  return kCocoClassNames[classId];
}
