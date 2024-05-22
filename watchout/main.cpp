#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <chrono>

#include <QApplication>
#include <opencv2/opencv.hpp>

#include "run_graph_main.h"
#include "face_parser.hpp"
#include "face_recognizer.hpp"
#include "mainwindow.h"
#include "common.hpp"

struct DMSResult {
	dms::GazeAngle gaze_angle;
	dms::EyeAspectRatio eye_aspect_ratio;
};

// hard code the graph content on `run_graph_main.cc` later
constexpr char graph_config_file[] = "/home/jetson/ssd/watchout/dependencies/mediapipe/mediapipe/graphs/iris_tracking/iris_tracking_gpu.pbtxt";

int authenticateDriver(int argc, char* argv[]) {
	QApplication auth_app(argc, argv);

	MainWindow auth_window;
	auth_window.setWindowState(Qt::WindowFullScreen);
	auth_window.show();

	return auth_app.exec();
}

void inferDriverStatus(
	dms::Pack<DMSLandmarks>& dmsl,
	dms::Pack<DMSResult>& dmsr,
	volatile bool& run) {
	std::this_thread::sleep_for(std::chrono::seconds(5));

	DMSLandmarks landmarks;
	dms::GazeAngle gaze_angle;
	dms::EyeAspectRatio eye_aspect_ratio;
	dms::GazeEstimator gaze_estimator;
	dms::EyeClosednessCalculator eye_closedness_calculator;
	dms::Rate rate(30);
	while (run) {
		{
			std::unique_lock<std::mutex> ul(dmsl.m);
			landmarks = dmsl();
		}

		gaze_angle = gaze_estimator.estimateGaze(landmarks, 640, 480);
		eye_aspect_ratio = eye_closedness_calculator.calculateEyeClosedness(landmarks);
		{
			std::unique_lock<std::mutex> ul(dmsr.m);
			dmsr().gaze_angle = gaze_angle;
			dmsr().eye_aspect_ratio = eye_aspect_ratio;
		}

		std::cout << __func__ << " " << rate.get() << " FPS" << std::endl;
		rate.sleep();
	}
}

int monitorDriver(int argc, char* argv[]) {
	MPPGraphRunnerWrapper dms_runner;
	dms_runner.initMPPGraph(graph_config_file);

	dms::Pack<DMSLandmarks> dms_landmarks;
	dms::Pack<DMSResult> dms_result;
	volatile bool run_inferrer = true;
	std::thread th_inferrer(inferDriverStatus, std::ref(dms_landmarks), std::ref(dms_result), std::ref(run_inferrer));

	cv::VideoCapture capture(0);
	// capture.set(cv::CAP_PROP_FRAME_WIDTH, 240);
	// capture.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
	cv::Mat input_frame;
	cv::Mat output_frame;
	DMSLandmarks landmarks;
	DMSResult result;

	// cv::namedWindow("Result", cv::WINDOW_NORMAL);
	// cv::setWindowProperty("Result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

	bool landmark_exists = false;
	bool run_landmarker = true;
	dms::Rate rate(100);
	while (run_landmarker) {
		capture.read(input_frame);
		cv::cvtColor(input_frame, input_frame, cv::COLOR_BGR2RGBA);
		size_t frame_timestamp = static_cast<double>(cv::getTickCount()) / static_cast<double>(cv::getTickFrequency()) * 1e6;
		dms_runner.processFrame(input_frame, frame_timestamp, output_frame, landmarks, landmark_exists);

		if (landmark_exists) {
			std::unique_lock<std::mutex> ul(dms_landmarks.m);
			dms_landmarks() = landmarks;
		}

		if (cv::waitKey(10) >= 0) {
			run_landmarker = false;
			run_inferrer = false;
		}

		{
			std::unique_lock<std::mutex> ul(dms_result.m);
			result = dms_result();
		}

		std::string caption(std::to_string(rate.get()) + " FPS | " + std::to_string(result.gaze_angle.yaw) + " | " + std::to_string(result.gaze_angle.pitch) + " | " + std::to_string(result.eye_aspect_ratio.ear));
		cv::putText(output_frame, caption, {10, 40}, 2, 1, {0, 0, 255});
		std::cout << caption << std::endl;
		cv::imshow("Result", output_frame);
	}

	capture.release();

	th_inferrer.join();

	return 0;
}

int runDMS(int argc, char* argv[]) {
	int auth_ret = authenticateDriver(argc, argv);
	int moni_ret = monitorDriver(argc, argv);

	return auth_ret | moni_ret; // ??
}

int main(int argc, char* argv[]) {
	return runDMS(argc, argv);
}
