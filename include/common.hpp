#pragma once

#include <string>
// #include dlib

namespace dms {
	/*
	               ### <--(카메라)
	                |
	           _____|_________________
	          /  ---|--> X [0.0, 1.0]/ <--(안면 이미지)
	         /  /   |               /
	        /  /    V              /
	       /  /     Z             / <--(Z값이 작을수록 카메라와 근접)
	      /  y [0.0, 1.0]        /     (X, Y와 스케일이 비슷)
	     /                      /
	    /______________________/

	*/
	//  **불확실**

	struct Point2D {
		float x;
		float y;
	};
    Point2D operator+(const Point2D& augend, const Point2D& addend) {
        return { augend.x + addend.x, augend.y + addend.y };
    }
    Point2D operator-(const Point2D& minuend, const Point2D& subtrahend) {
        return { minuend.x - subtrahend.x, minuend.y - subtrahend.y };
    }
	Point2D operator*(const Point2D& multiplicand, float multiplier) {
		return { multiplicand.x * multiplier, multiplicand.y * multiplier };
	}
	Point2D operator/(const Point2D& dividend, float divisor) {
		return { dividend.x / divisor, dividend.y / divisor };
	}

	struct Point3D {
		float x;
		float y;
		float z;
	};
    Point3D operator+(const Point3D& augend, const Point3D& addend) {
        return { augend.x + addend.x, augend.y + addend.y, augend.z + addend.z };
    }    
    Point3D operator-(const Point3D& minuend, const Point3D& subtrahend) {
        return { minuend.x - subtrahend.x, minuend.y - subtrahend.y, minuend.z - subtrahend.z };
    }
	Point3D operator*(const Point3D& multiplicand, float multiplier) {
		return { multiplicand.x * multiplier, multiplicand.y * multiplier, multiplicand.z * multiplier };
	}
	Point3D operator/(const Point3D& dividend, float divisor) {
		return { dividend.x / divisor, dividend.y / divisor, dividend.z / divisor };
	}

    struct Angle2D {
        float hor;
        float ver;
    };

    struct Angle3D {
        float roll;
        float pitch;
        float yaw;
    };

    struct GazeEstimatorCoefficients {
        Point2D coeffs[5];
    };

	struct EyeAspectRatio {
		float left;
		float right;
	};

	struct EyeGazeLandmarks {
		// medial canthal angle (MCA)
		Point3D mca_l; // MediaPipe landmark #362  (운전자의 좌측)
		Point3D mca_r; // MP 133                   (운전자의 우측)

		// midpoint between the eyes
		Point3D me; // MP 168

		// bottom of the nose
		Point3D bn; // MP 2

		// center of the right, top, left,
		// and bottom corners of the limbus
		// of irises
		// 안쪽에서부터 바깥쪽으로 돌아가는 방향
		// (운전자의 우안)        (운전자의 좌안)
		//      T[1]                  T[1]
		//       /\                    /\ 
		//      /  \ R[0] <---------- /  \ R[2] <--- (개발자의 우측)
		// L[2] \  / <---------- L[0] \  / <-------- (개발자의 좌측)
		//       \/                    \/
		//      B[3]                  B[3]
		//
		//                        [0]  [1]  [2]  [3]
		Point3D pupil_l[4]; // MP 476, 475, 474, 477    (운전자의 좌안)
		Point3D pupil_r[4]; // MP 469, 470, 471, 472    (운전자의 우안)
	};

	struct EyeClosednessLandmarks {
		// medial canthal angle (MCA)   [0]
		// mid inner and outer points
		// of the upper lid             [1], [2]
		// lateral canthal angle (LCA)  [3]
		// mid outer and inner points
		// of the lower lid             [4], [5]
		// 안쪽에서부터 바깥쪽으로 돌아가는 방향
		Point3D lid_l[6]; // MP 362, 384, 387, 263, 373, 381 (운전자의 좌안)
		Point3D lid_r[6]; // MP 133, 157, 160,  33, 144, 154 (운전자의 우안)
	};

    struct DriverInfo {
        const std::string name;
        // vector of dlib matrices
        GazeEstimatorCoefficients gec_hor;
        GazeEstimatorCoefficients gec_ver;
    };

    bool saveDriverInfo(const DriverInfo& di, int& err) {
        return true;
    }

    bool loadDriverInfo(DriverInfo& di, int& err) {
        return true;
    }

    // dlib::
	// 얼굴 인식 모델을 설계하기 위한 CNN 네트워크 블록 정의
	// Identity mapping 정의
	template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
	using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

	// Identity mapping 정의 (input, output channel과 resolution이 다를 때)
	template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
	using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

	// Conv layer 정의
	template <int N, template <typename> class BN, int stride, typename SUBNET>
	using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

	// Residual block 정의 (batch norm을 affine transform으로 대체)
	template <int N, typename SUBNET>
	using ares = relu<residual<block, N, affine, SUBNET>>;

	// Residual block 정의 (batch norm을 affine transform으로 대체)
	template <int N, typename SUBNET>
	using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

	// ResNet의 stage 정의
	template <typename SUBNET>
	using alevel0 = ares_down<256, SUBNET>;
	template <typename SUBNET>
	using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
	template <typename SUBNET>
	using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
	template <typename SUBNET>
	using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
	template <typename SUBNET>
	using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

	// ResNet 정의
	using anet_type = loss_metric<
	    fc_no_bias<
	        128, avg_pool_everything<
	                 alevel0<alevel1<alevel2<alevel3<alevel4<
	                     max_pool<
	                         3, 3, 2, 2,
	                         relu<affine<con<
	                             32, 7, 7, 2, 2,
	                             input_rgb_image_sized<150>>>>>>>>>>>>>;
}