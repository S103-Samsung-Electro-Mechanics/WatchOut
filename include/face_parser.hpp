/********************************
 * 1. Facial landmark의 좌표를 갖고 시선을 추정하는 알고리즘 (고개와 눈 모두)
 * 사용하는 랜드마크 좌표는 다음과 같음: 양안 안쪽 끝(두 개), 눈 사이, 코 아래 끝,
 * 검은자의 상하좌우 끝부분 네 개의 좌표(네 개씩 두 개)
 * For more details on the algorithm, please refer to: https://arxiv.org/pdf/2401.00406
 * 
 * 2. Facial landmark의 좌표를 갖고 눈 감음 정도를 추정하는 알고리즘
 * Paper link: https://www.mdpi.com/2079-9292/11/19/3183
 * 
 * 3. MediaPipe face landmark:
 * https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
 ********************************/

#pragma once

// #include Dlib
// #include OpenCV
#include <string>
#include <vector>
#include "common.hpp"

namespace dms {
	class EyeParser {
	private:
	public:
		EyeParser() {}
		Angle2D calcGazeDirection(const EyeGazeLandmarks& egl) {
		}
		EyeAspectRatio calcEAR(const EyeClosednessLandmarks& ecl) {}
	};
}