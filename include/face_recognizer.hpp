#pragma once

// #include Dlib
// #include OpenCV
#include <string>
#include <vector>

/*
이미지에서 가장 큰 얼굴을 잘라내 반환하는 클래스
*/

using namespace std;
using namespace cv;
using namespace dlib;

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

class DriverRegistrar {
private:
	const std::string SHAPE_PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat";
	const std::string FACE_RECOGNIZER_PATH = "dlib_face_recognition_resnet_model_v1.dat";

	frontal_face_detector detector;
	shape_predictor predictor;
	anet_type face_recognizer;

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
	    const vector<matrix<float, 0, 1>>& embedding_vectors,
	    int& err) {
	}

public:
	DriverRegistrar() : detector(get_frontal_face_detector()),
	                    predictor(deserialize(SHAPE_PREDICTOR_PATH).operator>>),
	                    face_recognizer(deserialize(FACE_RECOGNIZER_PATH).operator>>) {}

	/*
	bool registerDriver(
		vector<Mat>& main_cam_images,
		const string& driver_name,
		int& err)

	Description
	    운전자 안면 이미지들을 받아 임베딩 벡터들을 저장한다.
	    만약 전달 받은 이미지들에서 얼굴을 하나도 검출할 수 없다면 false를 반환한다.
	Argument:
	    1. vector<Mat>& main_cam_images: 운전자 감시 카메라에서 받은 이미지들.
	    2. const string& driver_name: 운전자 이름
	    3. int& err: 에러 코드 (추후 구현)
	Return:
	    등록 성공 여부
	*/
	bool registerDriver(vector<Mat>& main_cam_images, const string& driver_name, int& err) {
	}
};

class DriverAuthenticator {
private:
	const std::string SHAPE_PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat";
	const std::string FACE_RECOGNIZER_PATH = "dlib_face_recognition_resnet_model_v1.dat";

	frontal_face_detector detector;
	shape_predictor predictor;
	anet_type face_recognizer;

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
	    const string& filename,
	    vector<matrix<float, 0, 1>>& embedding_vectors,
	    int& err) {
	}

public:
	DriverAuthenticator() : detector(get_frontal_face_detector()),
	                        predictor(deserialize(SHAPE_PREDICTOR_PATH).operator>>),
	                        face_recognizer(deserialize(FACE_RECOGNIZER_PATH).operator>>) {}

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
	bool authenticateDriver(Mat& main_cam_image, string& driver_name, int& err) {

	}
};