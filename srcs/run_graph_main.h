// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#pragma once

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <string>

/*
 * Required landmarks:
 *
 * For gaze estimation:
 * 4, 152, 263, 33, 287, 57, 468, 473
 *
 * For E.A.R.:
 * 133, 158, 160, 33, 144, 153,
 * 362, 285, 387, 263, 373, 380
 */

enum LandmarkNames {
	NOSE_TIP = 0,                             // 4
	CHIN = 1,                                 // 152
	MOUTH_LEFT_CORNER = 2,                    // 287
	MOUTH_RIGHT_CORNER = 3,                   // 57
	LEFT_EYE_RIGHT_CORNER = 4,                // 362
	LEFT_EYE_UPPER_LID_MID_RIGHT_POINT = 5,   // 385
	LEFT_EYE_UPPER_LID_MID_LEFT_POINT = 6,    // 387
	LEFT_EYE_LEFT_CORNER = 7,                 // 263
	LEFT_EYE_LOWER_LID_MID_LEFT_POINT = 8,    // 373
	LEFT_EYE_LOWER_LID_MID_RIGHT_POINT = 9,   // 380
	RIGHT_EYE_LEFT_CORNER = 10,               // 133
	RIGHT_EYE_UPPER_LID_MID_LEFT_POINT = 11,  // 158
	RIGHT_EYE_UPPER_LID_MID_RIGHT_POINT = 12, // 160
	RIGHT_EYE_RIGHT_CORNER = 13,              // 33
	RIGHT_EYE_LOWER_LID_MID_RIGHT_POINT = 14, // 144
	RIGHT_EYE_LOWER_LID_MID_LEFT_POINT = 15,  // 153
	LEFT_PUPIL_CENTER = 16,                   // 473
	RIGHT_PUPIL_CENTER = 17                   // 468
};

const int landmark_converting_table[18]{4, 152, 287, 57, 362, 385, 387, 263, 373, 380, 133, 158, 160, 33, 144, 153, 473, 468};

struct DMSLandmarks {
	cv::Point3d landmarks[18];
};

class MPPGraphRunnerWrapper {
private:
	void* core_runner_ptr;

public:
	//   MPPGraphRunnerWrapper() {}
	~MPPGraphRunnerWrapper();
	bool initMPPGraph(std::string);
	bool processFrame(cv::Mat&, size_t, cv::Mat&, DMSLandmarks&, bool&);
};
