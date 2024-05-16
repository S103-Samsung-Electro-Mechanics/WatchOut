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
#include <cmath>
#include <string>
#include <vector>

#include "common.hpp"

namespace dms {
	class EyeParser {
	private:
		const GazeEstimatorCoefficients gec;

		inline Point2f calcHeadRotation(const EyeGazeLandmarks& egl) const {
			return {
			    std::atan((egl.me.z - egl.bn.z) / (egl.me.y - egl.bn.y)),            // ver (around x axis)
			    std::atan((egl.mca_l.z - egl.mca_r.z) / (egl.mca_l.x - egl.mca_r.x)) // hor (around y axis)
			};
		}

		inline Point2f calcRelativeLength(const EyeGazeLandmarks& egl) const {
			return {
			    std::sqrt(
			        std::pow(egl.me.x - egl.bn.x, 2.0f) +
			        std::pow(egl.me.y - egl.bn.y, 2.0f) +
			        std::pow(egl.me.z - egl.bn.z, 2.0f)), // relatvie width of the face to the image frame
			    std::sqrt(
			        std::pow(egl.mca_l.x - egl.mca_r.x, 2.0f) +
			        std::pow(egl.mca_l.y - egl.mca_r.y, 2.0f) +
			        std::pow(egl.mca_l.z - egl.mca_r.z, 2.0f)) // rel height of the face
			};
		}

		inline Point2f calcPupilCenter(
		    const EyeGazeLandmarks& egl,
		    const Point2f& face_rel_len) const {
			Point2f pupil_l{0.0f, 0.0f};
			Point2f pupil_r{0.0f, 0.0f};

			for (int i = 0; i < 4; ++i) {
				pupil_l.x -= egl.pupil_l[i].x;
				pupil_l.y -= egl.pupil_l[i].y;
				pupil_r.x -= egl.pupil_r[i].x;
				pupil_r.y -= egl.pupil_r[i].y;
			}

			pupil_l.x /= 4.0f;
			pupil_l.y /= 4.0f;
			pupil_r.x /= 4.0f;
			pupil_r.y /= 4.0f;

			pupil_l.x += egl.mca_l.x;
			pupil_l.y += egl.mca_l.y;
			pupil_r.x += egl.mca_r.x;
			pupil_r.y += egl.mca_r.y;

			pupil_l.x /= face_rel_len.x;
			pupil_l.y /= face_rel_len.y;
			pupil_r.x /= face_rel_len.x;
			pupil_r.y /= face_rel_len.y;

			return (pupil_l + pupil_r) / 2.0;
		}

		inline float calcLength(Point3f st, Point3f ed)
		{
		return std::sqrt(
			std::pow(st.x - ed.x, 2.0f) +
			std::pow(st.y - ed.y, 2.0f));
		}

	public:
		EyeParser(const GazeEstimatorCoefficients& gec) : gec(gec) {}

		Point2f calcGazeDirection(const EyeGazeLandmarks& egl) {
			Point2f gaze_dir{0.0f, 0.0f};
			Point2f rotation = calcHeadRotation(egl);
			Point2f face_rel_len = calcRelativeLength(egl);
			Point2f pupil = calcPupilCenter(egl, face_rel_len);
			Point2f face_center{egl.me.x, egl.me.y};

			gaze_dir.x = gec.coeffs[0].x;
			gaze_dir.y = gec.coeffs[0].y;

			gaze_dir.x += gec.coeffs[1].x * rotation.hor;
			gaze_dir.y += gec.coeffs[1].y * rotation.ver;

			gaze_dir.x += gec.coeffs[2].x * pupil.x;
			gaze_dir.y += gec.coeffs[2].y * pupil.y;

			gaze_dir.x += gec.coeffs[3].x * face_rel_len.x;
			gaze_dir.y += gec.coeffs[3].y * face_rel_len.y;

			gaze_dir.x += gec.coeffs[4].x * face_center.x;
			gaze_dir.y += gec.coeffs[4].y * face_center.y;

			return gaze_dir;
		}

		EyeAspectRatio calcEAR(const EyeClosednessLandmarks& ecl) {
			float eyelength1_l = calcLength(ecl.lid_l[1], ecl.lid_l[5]);  //1 5
			float eyelength2_l = calcLength(ecl.lid_l[2], ecl.lid_l[4]);  //2 4
			float eyewidth_l = calcLength(ecl.lid_l[0], ecl.lid_l[3]);  //가로 0 3

			float eyelength1_r = calcLength(ecl.lid_r[1], ecl.lid_r[5]);
			float eyelength2_r = calcLength(ecl.lid_r[2], ecl.lid_r[4]);
			float eyewidth_r = calcLength(ecl.lid_r[0], ecl.lid_r[3]);

			return {(eyelength1_l+eyelength2_l+eyelength1_r+eyelength2_r) / (eyewidth_l + eyewidth_r)};
		}
	};
}