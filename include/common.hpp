#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

namespace dms {
	using errcode = std::uint32_t;
	using embedding_vector = dlib::matrix<float, 0, 1>;

	template <typename _Ty>
	struct Pack {
		std::mutex m;
		_Ty val;
		_Ty& operator()() { return this->val; }
		_Ty& value() { return this->val; }
	};

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

	template <typename _Ty>
	struct Point2_ {
		_Ty x;
		_Ty y;
	};
	Point2_ operator+(const Point2_& left, const Point2_& right) {
		return {left.x + right.x, left.y + right.x};
	}
	Point2_ operator-(const Point2_& left, const Point2_& right) {
		return {left.x - right.x, left.y - right.y};
	}

	template <typename _Ty>
	struct Point3_ {
		_Ty x;
		_Ty y;
		_Ty z;
	};
	Point2_ operator+(const Point2_& left, const Point2_& right) {
		return {left.x + right.x, left.y + right.x, left.z + right.z};
	}
	Point2_ operator-(const Point2_& left, const Point2_& right) {
		return {left.x - right.x, left.y - right.y, left.z - right.z};
	}

	using Point2f = Point2_<float>;
	using Point3f = Point3_<float>;

	class Rate {
	private:
		std::chrono::steady_clock::time_point prev;
		std::chrono::steady_clock::time_point curr;
		std::chrono::steady_clock::time_point next;
		std::chrono::duration<std::uint32_t, std::nano> interval;
		std::chrono::duration<std::uint32_t, std::nano> span;

	public:
		Rate(const double rate) : prev(std::chrono::steady_clock::now()),
		                          curr(std::chrono::steady_clock::now()),
		                          next(std::chrono::steady_clock::now()),
		                          interval(static_cast<std::uint32_t>(1e9 / rate)),
		                          span(0) {}

		inline double get() {
			this->curr = std::chrono::steady_clock::now();
			this->span = std::chrono::duration_cast<std::chrono::nanoseconds>(this->curr - this->prev);
			this->prev = this->curr;
			
			return 1e9 / this->span.count();
		}

		inline void sleep() {
			std::this_thread::sleep_until(this->next);
			this->next = std::chrono::steady_clock::now() + this->interval;
		}
	};

	struct GazeEstimatorCoefficients {
		Point2f coeffs[5];
	};

	struct EyeAspectRatio {
		float left;
		float right;
	};

	struct EyeGazeLandmarks {
		// medial canthal angle (MCA)
		Point3f mca_l; // MediaPipe landmark #362  (운전자의 좌측)
		Point3f mca_r; // MP 133                   (운전자의 우측)

		// midpoint between the eyes
		Point3f me; // MP 168

		// bottom of the nose
		Point3f bn; // MP 2

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
		Point3f pupil_l[4]; // MP 476, 475, 474, 477    (운전자의 좌안)
		Point3f pupil_r[4]; // MP 469, 470, 471, 472    (운전자의 우안)
	};

	struct EyeClosednessLandmarks {
		// medial canthal angle (MCA)   [0]
		// mid inner and outer points
		// of the upper lid             [1], [2]
		// lateral canthal angle (LCA)  [3]
		// mid outer and inner points
		// of the lower lid             [4], [5]
		// 안쪽에서부터 바깥쪽으로 돌아가는 방향
		Point3f lid_l[6]; // MP 362, 384, 387, 263, 373, 381 (운전자의 좌안)
		Point3f lid_r[6]; // MP 133, 157, 160,  33, 144, 154 (운전자의 우안)
	};

	struct DriverInfo {
		const std::string name; // 1~4
		std::vector<dlib::matrix<float, 0, 1>> emb_vecs;
		GazeEstimatorCoefficients gec;
	};

	bool saveDriverInfo(const std::string& filename, const DriverInfo& di, int& err) {
		// di.emb_vecs[0] =
		ofstream ofs(filename, ios::binary);
		if (!ofs) {
			cerr << "Error: Unable to open file for writing." << endl;
			return false;
		}

		size_t num_embedding_vectors = di.emb_vecs.size();
		ofs.write(reinterpret_cast<const char*>(&num_embedding_vectors), sizeof(num_embedding_vectors));

		for (const auto& descriptor : di.emb_vecs) {
			size_t num_rows = descriptor.nr();
			size_t num_cols = descriptor.nc();

			ofs.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
			ofs.write(reinterpret_cast<const char*>(&num_cols), sizeof(num_cols));
			ofs.write(reinterpret_cast<const char*>(descriptor.begin()), num_rows * num_cols * sizeof(float));
		}
		// Save GazeEstimatorCoefficients
		for (int i = 0; i < 5; ++i) {
			ofs.write(reinterpret_cast<const char*>(&di.gec.coeffs[i].x), sizeof(float));
			ofs.write(reinterpret_cast<const char*>(&di.gec.coeffs[i].y), sizeof(float));
		}
		ofs.close();
		return true;
	}

	bool loadDriverInfo(const std::string& filename, const DriverInfo& di, int& err) {
		ifstream ifs(filename, ios::binary);
		if (!ifs) {
			cerr << "Error: Unable to open file for reading." << endl;
			return false;
		}

		size_t num_embedding_vectors;
		ifs.read(reinterpret_cast<char*>(&num_embedding_vectors), sizeof(num_embedding_vectors));

		di.emb_vecs.resize(num_embedding_vectors);

		for (auto& descriptor : di.emb_vecs) {
			size_t num_rows, num_cols;
			ifs.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
			ifs.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

			descriptor.set_size(num_rows, num_cols);
			ifs.read(reinterpret_cast<char*>(descriptor.begin()), num_rows * num_cols * sizeof(float));
		}

		// Load GazeEstimatorCoefficients
		for (int i = 0; i < 5; ++i) {
			ifs.read(reinterpret_cast<char*>(&di.gec.coeffs[i].x), sizeof(float));
			ifs.read(reinterpret_cast<char*>(&di.gec.coeffs[i].y), sizeof(float));
		}

		ifs.close();
		return true;
	}

	// 얼굴 인식 모델을 설계하기 위한 CNN 네트워크 블록 정의
	// Identity mapping 정의
	template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
	using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

	// Identity mapping 정의 (input, output channel과 resolution이 다를 때)
	template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
	using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

	// Conv layer 정의
	template <int N, template <typename> class BN, int stride, typename SUBNET>
	using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

	// Residual block 정의 (batch norm을 affine transform으로 대체)
	template <int N, typename SUBNET>
	using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;

	// Residual block 정의 (batch norm을 affine transform으로 대체)
	template <int N, typename SUBNET>
	using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

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
	using anet_type = dlib::loss_metric<
	    dlib::fc_no_bias<
	        128, dlib::avg_pool_everything<
	                 alevel0<alevel1<alevel2<alevel3<alevel4<
	                     dlib::max_pool<
	                         3, 3, 2, 2,
	                         dlib::relu<dlib::affine<dlib::con<
	                             32, 7, 7, 2, 2,
	                             dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;
}