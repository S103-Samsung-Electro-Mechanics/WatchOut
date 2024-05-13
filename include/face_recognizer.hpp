#pragma once

// #include Dlib
// #include OpenCV
#include <string>
#include <vector>
#include "common.hpp"

/*
이미지에서 가장 큰 얼굴을 잘라내 반환하는 클래스
*/

namespace dms {
	class DriverRegistrar {
	private:
		const std::string SHAPE_PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat";
		const std::string FACE_RECOGNIZER_PATH = "dlib_face_recognition_resnet_model_v1.dat";

		dlib::frontal_face_detector detector;
		dlib::shape_predictor predictor;
		dms::anet_type face_recognizer;

		/*
		void saveEmbeddingVector(
		    const string& filename,
		    const vector<matrix>& embedding_vectors)
		Description
		    embedding_vectors를 filename에 바이너리로 저장한다.
		Argument
		    1. const string& filename: 파일 경로
		    2. const vector<matrix<float, 0, 1>>& embedding_vectors:
		        운전자 안면 임베딩 벡터
		    3. int& err: 에러 코드 (추후 구현)
		Return
		*/
		void saveEmbeddingVector(
		    const string& filename,
		    const std::vector<dms::matrix<float, 0, 1>>& embedding_vectors,
		    int& err) {
		}

	public:
		DriverRegistrar() : detector(dlib::get_frontal_face_detector()),
		                    predictor(dlib::deserialize(SHAPE_PREDICTOR_PATH).operator>>),
		                    face_recognizer(dlib::deserialize(FACE_RECOGNIZER_PATH).operator>>) {}

		/*
		bool registerDriver(
		    vector<Mat>& main_cam_images,
			cosnt vector<Angle2D>& driver_gaze_angle,
		    const string& driver_name,
		    int& err)

		Description
		    운전자 안면 이미지들을 받아 임베딩 벡터들을 저장한다.
		    만약 전달 받은 이미지들에서 얼굴을 하나도 검출할 수 없다면 false를 반환한다.
		Argument:
		    1. vector<Mat>& main_cam_images: 운전자 감시 카메라에서 받은 이미지들.
			2. const vector<Angle2D>& driver_gaze_angle: 운전자가 main_cam_image
			   상에서의 시선 각도. main_cam_images와 size()가 같아야 함.
		    3. const string& driver_name: 운전자 이름
		    4. int& err: 에러 코드 (추후 구현)
		Return:
		    등록 성공 여부
		*/
		bool registerDriver(
			std::vector<cv::Mat>& main_cam_images,
			const std::vector<Angle2D>& driver_gaze_angle,
			const std::string& driver_name,
			int& err) {
		}
	};

	class DriverAuthenticator {
	private:
		const std::string SHAPE_PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat";
		const std::string FACE_RECOGNIZER_PATH = "dlib_face_recognition_resnet_model_v1.dat";

		dlib::frontal_face_detector detector;
		dlib::shape_predictor predictor;
		dms::anet_type face_recognizer;

		/*
		void loadEmbeddingVector(
		    const string& filename,
		    vector<matrix>& embedding_vectors)
		Description
		    filename에 저장된 임베딩 벡터들을 embedding_vectors에 로드한다.
		Argument
		    1. const string& filename: 파일 경로
		    2. vector<matrix<float, 0, 1>>& embedding_vectors:
		        운전자 안면 임베딩 벡터
		    3. int& err: 에러 코드 (추후 구현)
		Return
		*/
		void loadEmbeddingVector(
		    const std::string& filename,
		    std::vector<dms::matrix<float, 0, 1>>& embedding_vectors,
		    int& err) {
		}

	public:
		DriverAuthenticator() : detector(dlib::get_frontal_face_detector()),
		                        predictor(dlib::deserialize(SHAPE_PREDICTOR_PATH).operator>>),
		                        face_recognizer(dlib::deserialize(FACE_RECOGNIZER_PATH).operator>>) {}

		/*
		bool authenticateDriver(
		    Mat& main_cam_image,
		    string& driver_name,
		    int& err)

		Description
		    * 현재 운전석에 앉아 있는 운전자의 이미지를 받는다.
		    * 디스크에 저장된 등록된 운전자들의 안면 임베딩 벡터들과 비교해 등록된 운전자인지를 파악한다.
		        벡터거리배열
		        for 운전자 in 모든등록운전자:
		            for 안면벡터 in 안면벡터들:
		                벡터거리평균 += 거리(안면벡터, 현재 이미지 안면벡터)

		            벡터거리평균 /= 해당 운전자의 저장된 안면벡터 개수

		            벡터거리배열.append(벡터거리평균)

		        최소거리인덱스 = 최소인덱스(벡터거리배열)

		        if 벡터거리배열[최소거리인덱스] < 문턱값:
		            return true;
		        else:
		            return false;
		Argument:
		    1. Mat& main_cam_image: 운전자 감시 카메라에서 받은 이미지
		    2. string& driver_name 인식된 운전자 이름
		    3. int& err: 에러 코드 (추후 구현)
		Return:
		    인증 성공 여부
		*/
		bool authenticateDriver(cv::Mat& main_cam_image, std::string& driver_name, int& err) {
		}
	};
}
