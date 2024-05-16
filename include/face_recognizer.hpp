#pragma once

// #include Dlib
// #include OpenCV
#include <string>
#include <vector>

#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>

#include "common.hpp"

namespace dms {
	class DriverRegistrar {
	private:
		const std::string SHAPE_PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat";
		const std::string FACE_RECOGNIZER_PATH = "dlib_face_recognition_resnet_model_v1.dat";

		dlib::frontal_face_detector detector;
		dlib::shape_predictor predictor;
		dms::anet_type face_recognizer;

	public:
		DriverRegistrar() : detector(dlib::get_frontal_face_detector()),
		                    predictor(),
		                    face_recognizer() {
			dlib::deserialize(SHAPE_PREDICTOR_PATH) >> predictor;
			dlib::deserialize(FACE_RECOGNIZER_PATH) >> face_recognizer;
		}

		bool registerDriver(
		    std::vector<cv::Mat>& main_cam_images,
		    const std::vector<Point2f>& driver_gaze_angle,
		    const std::string& driver_name,
		    int& err) {
			DriverInfo driver_info;
			std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
			std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
			driver_info.name = driver_name;

			for (auto cam_image : main_cam_images) {
				// main_cam_images를 matrix<rgb_pixel>로 변환
				dlib::matrix<dlib::rgb_pixel> rgb_image;
				dlib::assign_image(rgb_image, dlib::cv_image<dlib::bgr_pixel>(cam_image)); // rgb_pixel로 변경
				//***********************
				// 가장 큰 얼굴 찾아서 넣기
				//***********************
				for (auto face : this->detector(rgb_image)) {
					auto shape = this->predictor(rgb_image, face);
					dlib::matrix<dlib::rgb_pixel> face_chip;
					// 이미지에서 얼굴 탐지기를 실행하고 각 얼굴에 대해 150x150 픽셀 크기로 정규화되고 적절하게 회전되고 중앙에 맞도록 복사본을 추출합니다.
					dlib::extract_image_chip(rgb_image, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
					// 이미지에서 얼굴 찾고 faces에 푸시
					faces.push_back(std::move(face_chip));
				}
			}
			// faces size()가 3 이하 return false
			if (faces.size() <= 3) return false;
			// 128vector 로 전환
			driver_info.emb_vecs = this->face_recognizer(faces);

			// saveEmbeddingVector(); 이름 1 ~ 4 숫자로 우선 할까..?

			//////////////////////////
			// driver_info.gec_hor =
			// driver_info.gec_ver =
			//////////////////////////

			std::string str1 = "./drivers/";
			std::string str2 = ".bin";
			std::string path = str1 + driver_name + str2;

			if (!saveDriverInfo(path, driver_info, err)) { // driver_info에 임베딩 벡터 넣어야 함
				                                           // error parsing
			}
			else return true;
		}
	};

	class DriverAuthenticator {
	private:
		const std::string SHAPE_PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat";
		const std::string FACE_RECOGNIZER_PATH = "dlib_face_recognition_resnet_model_v1.dat";

		dlib::frontal_face_detector detector;
		dlib::shape_predictor predictor;
		dms::anet_type face_recognizer;

	public:
		DriverAuthenticator() : detector(dlib::get_frontal_face_detector()),
		                        predictor(),
		                        face_recognizer() {
			dlib::deserialize(SHAPE_PREDICTOR_PATH) >> predictor;
			dlib::deserialize(FACE_RECOGNIZER_PATH) >> face_recognizer;
		}

		bool authenticateDriver(cv::Mat& main_cam_image, std::string& driver_name, int& err) {
			DriverInfo driver_info[4];
			std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
			// 이미지 matrix<RGB_pixel>로 변환
			dlib::matrix<dlib::rgb_pixel> driver_img;
			dlib::assign_image(driver_img, dlib::cv_image<dlib::bgr_pixel>(main_cam_image)); // rgb_pixel로 변경
			// ****************************************
			// 제일 큰 이미지 하나만 faces에 푸시해야함 <- 이거 없음
			// ****************************************
			for (auto face : this->detector(driver_img)) {
				auto shape = this->predictor(driver_img, face);
				dlib::matrix<dlib::rgb_pixel> face_chip;
				dlib::extract_image_chip(driver_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(std::move(face_chip));
			}
			if (faces.size() != 1) {
				std::cout << faces.size() << "명 얼굴이 인식되었습니다" << std::endl;
				// qt 에서 띄우는걸로 바꿔야함 (err로 가져가서 main에서 띄워야할듯)
				std::cout << "한명 얼굴 인식하라고 ㅡㅡ" << std::endl;
				return false;
			}
			// 128 vector 변환
			dlib::matrix<float, 0, 1> driver_descriptor = this->face_recognizer(faces[0]);
			// 디스크 저장된 등록된 운전자 벡터 받아오기
			bool DAT_flag[4]; // 안에 정보가 있는지 체크

			////
			std::vector<bool> dat_load(4);
			////
			for (int i = 0; i < 4; i++) {
				std::string str1 = "./drivers/";
				std::string str2 = ".bin";
				std::string path = str1 + std::to_string(i + 1) + str2;
				int err;

				if (!loadDriverInfo(path, driver_info[i], err)) {
					// error parsing
					dat_load[i] = false; // ?????
				}
				else {
					dat_load[i] = true;
				}
				if (!dat_load[i])
					std::cout << i + 1 << "번째 정보는 비어있습니다" << std::endl;
			}
			// 등록된 벡터와 비교
			float vector_lengths[4];
			int min_idx = 0;
			float sum_length = 0.0f;
			for (int i = 0; i < 4; i++) {
				if (!dat_load[i]) {
					vector_lengths[i] = 1;
					continue; // 만약 정보가 없는 idx면 패스
				}
				sum_length = 0;
				for (int j = 0; j < driver_info[i].emb_vecs.size(); ++j) {
					float leng = length(driver_info[i].emb_vecs[j] - driver_descriptor);
					std::cout << leng << '\n';
					sum_length += leng;
				}
				vector_lengths[i] = sum_length / driver_info[i].emb_vecs.size();
				if (vector_lengths[i] < vector_lengths[min_idx]) {
					min_idx = i;
				}
			}
			if (vector_lengths[min_idx] < 0.45) {
				driver_name = driver_info[min_idx].name;
				return true; // 인증 성공
			}

			return false;
		}
	};
}
