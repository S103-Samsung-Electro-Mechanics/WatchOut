#pragma once
#ifndef GAZE_TRACKING_HPP
#define GAZE_TRACKING_HPP

#include "common.hpp"
#include "run_graph_main.h"
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#define PI 3.14159265358979323846

namespace dms {
	class GazeEstimator {
	  private:
		cv::Point2d relative(const cv::Point3d point, const size_t frame_width, const size_t frame_height) {
			return cv::Point2d(static_cast<int>(point.x * frame_width), static_cast<int>(point.y * frame_height));
		}

		cv::Point3d relativeT(const cv::Point3d point, const size_t frame_width, const size_t frame_height) {
			return cv::Point3d(static_cast<int>(point.x * frame_width), static_cast<int>(point.y * frame_height), 0);
		}

	  public:
		GazeAngle estimateGaze(const DMSLandmarks& dmsl, const size_t frame_width, const size_t frame_height) {
			// 2D image points
			std::vector<cv::Point2d> image_points = {
			    relative(dmsl.landmarks[0], frame_width, frame_height),  // Nose tip
			    relative(dmsl.landmarks[1], frame_width, frame_height),  // Chin
			    relative(dmsl.landmarks[7], frame_width, frame_height),  // Left eye left corner
			    relative(dmsl.landmarks[13], frame_width, frame_height), // Right eye right corner
			    relative(dmsl.landmarks[2], frame_width, frame_height),  // Left Mouth corner
			    relative(dmsl.landmarks[3], frame_width, frame_height)   // Right mouth corner
			};

			// general face 3D model points
			std::vector<cv::Point3d> model_points = {
			    cv::Point3d(0.0, 0.0, 0.0),       // Nose tip
			    cv::Point3d(0, -63.6, -12.5),     // Chin
			    cv::Point3d(-43.3, 32.7, -26),    // Left eye, left corner
			    cv::Point3d(43.3, 32.7, -26),     // Right eye, right corner
			    cv::Point3d(-28.9, -28.9, -24.1), // Left Mouth corner
			    cv::Point3d(28.9, -28.9, -24.1)   // Right mouth corner
			};

			cv::Point3d Eye_ball_center_right = cv::Point3d(29.05, 32.7, -39.5);
			cv::Point3d Eye_ball_center_left = cv::Point3d(-29.05, 32.7, -39.5);

			// Camera matrix estimation
			double focal_length = frame_width;
			cv::Point2d center(frame_width / 2, frame_height / 2);
			cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
			                         0, focal_length, center.y,
			                         0, 0, 1);

			cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

			// Solve PnP problem
			cv::Mat rotation_vector, translation_vector;
			cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

			// Calculate Head rotation vector
			cv::Mat rotation_matrix;
			cv::Rodrigues(rotation_vector, rotation_matrix);

			double sy = sqrt(rotation_matrix.at<double>(0, 0) * rotation_matrix.at<double>(0, 0) +
			                 rotation_matrix.at<double>(1, 0) * rotation_matrix.at<double>(1, 0));

			double x = 0, y = 0, z = 0;
			if (sy < 1e-6) {
				x = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2));
				y = atan2(-rotation_matrix.at<double>(2, 0), sy);
				z = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0));
			}
			else {
				x = atan2(-rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(1, 1));
				y = atan2(-rotation_matrix.at<double>(2, 0), sy);
				z = 0;
			}

			double head_pitch = x * 180 / PI;
			double head_yaw = y * 180 / PI;
			double head_roll = z * 180 / PI;

			// Pupil locations
			cv::Point2d left_pupil = relative(dmsl.landmarks[17], frame_width, frame_height);
			cv::Point2d right_pupil = relative(dmsl.landmarks[16], frame_width, frame_height);

			// Transformation between image point to world point
			cv::Mat transformation;

			std::vector<cv::Point3d> image_points1 = {
			    relativeT(dmsl.landmarks[0], frame_width, frame_height),  // Nose tip
			    relativeT(dmsl.landmarks[1], frame_width, frame_height),  // Chin
			    relativeT(dmsl.landmarks[7], frame_width, frame_height),  // Left eye left corner
			    relativeT(dmsl.landmarks[13], frame_width, frame_height), // Right eye right corner
			    relativeT(dmsl.landmarks[2], frame_width, frame_height),  // Left Mouth corner
			    relativeT(dmsl.landmarks[3], frame_width, frame_height)   // Right mouth corner
			};
			cv::estimateAffine3D(image_points1, model_points, transformation, cv::noArray());

			cv::Mat pupil_world_cord;
			cv::Mat S;

			if (!transformation.empty()) {
				// Project pupil image point into 3D world point
				cv::Mat pupil_world_cord_left = transformation * (cv::Mat_<double>(4, 1) << left_pupil.x, left_pupil.y, 0, 1);
				cv::Mat pupil_world_cord_right = transformation * (cv::Mat_<double>(4, 1) << right_pupil.x, right_pupil.y, 0, 1);

				// 3D gaze point
				cv::Mat S_left = (cv::Mat(Eye_ball_center_left) + (pupil_world_cord_left - cv::Mat(Eye_ball_center_left)) * 10);
				cv::Mat S_right = (cv::Mat(Eye_ball_center_right) + (pupil_world_cord_right - cv::Mat(Eye_ball_center_right)) * 10);

				pupil_world_cord = (pupil_world_cord_left + pupil_world_cord_right) * 0.5;
				S = (S_left + S_right) * 0.5;
			}
			else {
				if (head_yaw > 0) { // if turn right
					std::vector<cv::Point3d> image_points_left = {image_points1[0], image_points1[1], image_points1[2], image_points1[4]};
					std::vector<cv::Point3d> model_points_left = {model_points[0], model_points[1], model_points[2], model_points[4]};
					cv::estimateAffine3D(image_points_left, model_points_left, transformation, cv::noArray());
					pupil_world_cord = transformation * (cv::Mat_<double>(4, 1) << left_pupil.x, left_pupil.y, 0, 1);
					S = (cv::Mat(Eye_ball_center_left) + (pupil_world_cord - cv::Mat(Eye_ball_center_left)) * 10);
				}
				else { // if turn left
					std::vector<cv::Point3d> image_points_right = {image_points1[0], image_points1[1], image_points1[3], image_points1[5]};
					std::vector<cv::Point3d> model_points_right = {model_points[0], model_points[1], model_points[3], model_points[5]};
					cv::estimateAffine3D(image_points_right, model_points_right, transformation, cv::noArray());
					pupil_world_cord = transformation * (cv::Mat_<double>(4, 1) << right_pupil.x, right_pupil.y, 0, 1);
					S = (cv::Mat(Eye_ball_center_right) + (pupil_world_cord - cv::Mat(Eye_ball_center_right)) * 10);
				}
			}
			cv::Mat gaze_vector = S - pupil_world_cord;
			double gaze_yaw = atan2(gaze_vector.at<double>(0, 0), gaze_vector.at<double>(2, 0)) * 180 / PI;
			double gaze_pitch = atan2(gaze_vector.at<double>(1, 0), gaze_vector.at<double>(2, 0)) * 180 / PI;

			return {gaze_yaw, gaze_pitch};
		}
	};

	class EyeClosednessCalculator {
	  private:
	  public:
		EyeAspectRatio calculateEyeClosedness(const DMSLandmarks& dmsl) {
			double eyelength1_l = cv::norm(dmsl.landmarks[5] - dmsl.landmarks[9]); // 1 5
			double eyelength2_l = cv::norm(dmsl.landmarks[6] - dmsl.landmarks[8]); // 2 4
			double eyewidth_l = cv::norm(dmsl.landmarks[4] - dmsl.landmarks[7]);   // 가로 0 3

			double eyelength1_r = cv::norm(dmsl.landmarks[11] - dmsl.landmarks[15]); // 1 5
			double eyelength2_r = cv::norm(dmsl.landmarks[12] - dmsl.landmarks[14]); // 2 4
			double eyewidth_r = cv::norm(dmsl.landmarks[10] - dmsl.landmarks[13]);   // 가로 0 3

			return {(eyelength1_l + eyelength2_l + eyelength1_r + eyelength2_r) / (eyewidth_l + eyewidth_r)};
		}
	};
}

#endif