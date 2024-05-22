#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <chrono>
#include <mutex>

#include <dlib/dnn.h>
#include <dlib/matrix.h>

#include "run_graph_main.h"

namespace dms {
	// Bind a mutex variable with any type of data
	template <typename _Ty>
	struct Pack {
		std::mutex m;
		_Ty v;
		_Ty& operator()() { return this->v; }
		_Ty& value() { return this->v; }
	};

	class Rate {
	/*
	This class is a tool for syncing loop and calculate
	its iteration rate.

	You can instantiate an object of this class by setting
	desired iteration rate (iterations per second).
	*/
	private:
		std::chrono::steady_clock::time_point prev;
		std::chrono::steady_clock::time_point curr;
		std::chrono::steady_clock::time_point next;
		std::chrono::duration<std::uint32_t, std::nano> interval;
		std::chrono::duration<std::uint32_t, std::nano> span;

	public:
		/*
		Desired iteration rate must be provided
		*/
		Rate(const double rate) : prev(std::chrono::steady_clock::now()),
                             curr(std::chrono::steady_clock::now()),
                             next(std::chrono::steady_clock::now()),
                             interval(static_cast<std::uint32_t>(1e9 / rate)),
                             span(0) {}

		/*
		Calculates iteration rate. This method must be called
		only once in every iteration.
		*/
		inline double get();

		/*
		Sleep for minimum required amount of time to meet the
		rate requirement. If the loop execution timing already
		exceeded the requirement, this method does nothing.
		This method also must be called only once per loop.
		*/
		inline void sleep();
	};

	inline double Rate::get() {
		this->curr = std::chrono::steady_clock::now();
		this->span = std::chrono::duration_cast<std::chrono::nanoseconds>(this->curr - this->prev);
		this->prev = this->curr;

		return 1e9 / this->span.count();
	}

	inline void Rate::sleep() {
		std::this_thread::sleep_until(this->next);
		this->next = std::chrono::steady_clock::now() + this->interval;
	}

	struct GazeAngle {
		/*
		Custom data type to hold driver's gaze. `yaw` and
		`pitch` represent the angle of the gaze around the
		z and x axes respectively, given that x axis points
		towards to the right side of the head, while y axis
		points towards downwards.
		*/
		double yaw;   // (-) <-- LEFT -- 0 -- RIGHT --> (+)
		double pitch; // (-) <-- UP -- 0 -- DOWN --> (+)
	};

	struct EyeAspectRatio {
		/*
		Eye height : Eye width
		The higher the more eye is opened.
		*/
		double ear;
	};

	struct DriverInfo {
		/*
		Informations of a driver to save on a disk.
		`emb_vecs` represents the 128-element 1D embedding vector
		of a driver. Usually more than 3 to 4 facial embedding
		vectors are saved for better accuracy.
		*/
		std::string name;
		std::vector<dlib::matrix<float, 0, 1>> emb_vecs;
	};

	/*
	A template class definition of a neural net that transforms
	a facial image into an embedding vector.
	The network is ResNet-based.
	*/
	// Identity mapping
	template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
	using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

	// Identity mapping (when # of input channel != # of output channel and in resolution != out resolution)
	template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
	using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

	// Conv layer
	template <int N, template <typename> class BN, int stride, typename SUBNET>
	using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

	// Residual block
	template <int N, typename SUBNET>
	using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;

	// Residual block (batch norm -> affine transform)
	template <int N, typename SUBNET>
	using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

	// Stages of ResNet
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

	// ResNet
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

#endif