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
#include <cmath>

#include "common.hpp"

namespace dms {
	class EyeParser {
	private:
		const GazeEstimatorCoefficients gec;

		inline Angle2D calcHeadRotation(const EyeGazeLandmarks& egl) const {
			return {
				std::atan((egl.mca_l.z - egl.mca_r.z) / (egl.mca_l.x - egl.mca_r.x)), // hor (around y axis)
				std::atan((egl.me.z - egl.bn.z) / (egl.me.y - egl.bn.y)) // ver (around x axis)
			};
		}

		inline Point2D calcRelativeLength(const EyeGazeLandmarks& egl) const {
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

		inline Point2D calcPupilCenter(
			const EyeGazeLandmarks& egl,
			const Point2D& face_rel_len) const {
			Point2D pupil_l { 0.0f, 0.0f };
			Point2D pupil_r { 0.0f, 0.0f };

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

	public:
		EyeParser(const GazeEstimatorCoefficients& gec) : gec(gec) {}
		
		Angle2D calcGazeDirection(const EyeGazeLandmarks& egl) {
			Angle2D gd { 0.0f, 0.0f };
			Angle2D rotation = calcHeadRotation(egl);
			Point2D face_rel_len = calcRelativeLength(egl);
			Point2D pupil = calcPupilCenter(egl, face_rel_len);
			Point2D face_center { egl.me.x, egl.me.y };

			gd.hor = gec.coeffs[0].x;
			gd.ver = gec.coeffs[0].y;

			gd.hor += gec.coeffs[1].x * rotation.hor;
			gd.ver += gec.coeffs[1].y * rotation.ver;

			gd.hor += gec.coeffs[2].x * pupil.x;
			gd.ver += gec.coeffs[2].y * pupil.y;

			gd.hor += gec.coeffs[3].x * face_rel_len.x;
			gd.ver += gec.coeffs[3].y * face_rel_len.y;

			gd.hor += gec.coeffs[4].x * face_center.x;
			gd.ver += gec.coeffs[4].y * face_center.y;

			return gd;
		}

		EyeAspectRatio calcEAR(const EyeClosednessLandmarks& ecl) {
			return { 0.0f, 0.0f };
		}
	};
}