#ifndef FACE_RECOGNIZER_HPP
#define FACE_RECOGNIZER_HPP

#include <string>
#include <vector>

#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "common.hpp"

namespace dms {
	class DriverRegistrar {
		/*
		Register driver by saving the driver's facial embedding vector.
		*/
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

		/*
		Takes a facial images in a form of cv::Mat, passes them through
		a DNN network, and save their embedding vectors on a disk.
		*/
		bool registerDriver(std::vector<cv::Mat>& main_cam_images,
			const std::vector<cv::Point2d>& driver_gaze_angle,
			const std::string& driver_name,
			int& err)  {
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
			driver_info.emb_vecs = face_recognizer(faces);

			DriverInfo driver_trash;
			
			std::string str1 = "../drivers/";
			std::string str2 = ".bin";
			std::string path = str1 + "1" + str2;
			// int err;
			int err_save;
			/////////////////////////////////수정부분

			std::ofstream ofs(path, std::ios::binary);
			if (!ofs) {
				std::cerr << "Error: Unable to open file for writing." << std::endl;
				//error parsing
				return false;
			}

			size_t num_embedding_vectors = driver_info.emb_vecs.size();
			ofs.write(reinterpret_cast<const char*>(&num_embedding_vectors), sizeof(num_embedding_vectors));

			for (const auto& descriptor : driver_info.emb_vecs) {
				size_t num_rows = descriptor.nr();
				size_t num_cols = descriptor.nc();

				ofs.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
				ofs.write(reinterpret_cast<const char*>(&num_cols), sizeof(num_cols));
				ofs.write(reinterpret_cast<const char*>(descriptor.begin()), num_rows * num_cols * sizeof(float));
			}

			ofs.close();
			return true;
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

		/*
		Takes a single image and compare its embedding vector with all
		of the embedding vectors saved on the disk. Find out the most
		relevant--meaning nearest in terms of Euclidean distance--embedding
		vector, and verify if the distance is lower than the predefined
		threshold.
		*/
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

            bool dat_load[4];
            ////
            for (int i = 0; i < 4; i++) {
                std::string str1 = "../drivers/";
                std::string str2 = ".bin";
                std::string path = str1 + std::to_string(i + 1) + str2;
                // int err;
                
                std::ifstream ifs(path, std::ios::binary);
                if (!ifs) {
                    std::cerr << "Error: Unable to open file for reading." << std::endl;
                    dat_load[i] = false;
                }
                else {
                    size_t num_embedding_vectors;
                    ifs.read(reinterpret_cast<char*>(&num_embedding_vectors), sizeof(num_embedding_vectors));

                    driver_info[i].emb_vecs.resize(num_embedding_vectors);

                    for (auto& descriptor : driver_info[i].emb_vecs) {
                        size_t num_rows, num_cols;
                        ifs.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
                        ifs.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

                        descriptor.set_size(num_rows, num_cols);
                        ifs.read(reinterpret_cast<char*>(descriptor.begin()), num_rows * num_cols * sizeof(float));
                    }

                    ifs.close();
                    dat_load[i] = true;
                }
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
            if (vector_lengths[min_idx] == 1) {
                // 등록된사람 없습니다
                std::cout << "등록된 사람이 없습니다." << std::endl;
                return false;
            }
            // 인증된 사람이 아닙니다
            std::cout << "인증된 사람 아닙니다." << std::endl;

            return false;
        }
	};
}

#endif