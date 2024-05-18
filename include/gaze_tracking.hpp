
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

/**
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/resource_util.h"
#include <time.h>
#include <stdio.h>
#include <chrono>

#define PI 3.14159265358979323846
// =================== fix ============
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
// ===================================
*/

namespace dms
{
    class Gaze
    {
    private:
        float gaze_yaw;   // Left & Right
        float gaze_pitch; // Up & Down

    public:
        cv::Point2d relative(const mediapipe::NormalizedLandmark &landmark, const cv::Size &frameSize)
        {
            return cv::Point2d(static_cast<int>(landmark.x() * frameSize.width), static_cast<int>(landmark.y() * frameSize.height));
        }

        cv::Point3d relativeT(const mediapipe::NormalizedLandmark &landmark, const cv::Size &frameSize)
        {
            return cv::Point3d(static_cast<int>(landmark.x() * frameSize.width), static_cast<int>(landmark.y() * frameSize.height), 0);
        }

        void calcGaze(cv::Mat &frame, const mediapipe::NormalizedLandmarkList &points)
        {
            // 2D image points
            std::vector<cv::Point2d> image_points = {
                relative(points.landmark(4), frame.size()),   // Nose tip
                relative(points.landmark(152), frame.size()), // Chin
                relative(points.landmark(263), frame.size()), // Left eye left corner
                relative(points.landmark(33), frame.size()),  // Right eye right corner
                relative(points.landmark(287), frame.size()), // Left Mouth corner
                relative(points.landmark(57), frame.size())   // Right mouth corner
            };

            // std::cout << "idx : 4 x : " << points.landmark(4).x() << " y : " << points.landmark(4).y() << "\n";

            // 3D model points
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
            double focal_length = frame.cols;
            cv::Point2d center(frame.cols / 2, frame.rows / 2);
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
            if (sy < 1e-6)
            {
                x = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2));
                y = atan2(-rotation_matrix.at<double>(2, 0), sy);
                z = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0));
            }
            else
            {
                x = atan2(-rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(1, 1));
                y = atan2(-rotation_matrix.at<double>(2, 0), sy);
                z = 0;
            }

            double pitch = x * 180 / PI;
            double yaw = y * 180 / PI;
            double roll = z * 180 / PI;

            // Pupil locations
            cv::Point2d left_pupil = relative(points.landmark(468), frame.size());
            // cv::Point2d left_pupil = cv::Point2d(points.landmark(468).x()*640,points.landmark(468).y()*480);
            cv::Point2d right_pupil = relative(points.landmark(473), frame.size());
            // std::cout<<"left : "<<left_pupil.x <<" y : "<<left_pupil.y<<"\n";
            std::cout << "p4" << std::endl;
            // Transformation between image point to world point
            cv::Mat transformation;

            std::vector<cv::Point3d> image_points1 = {
                relativeT(points.landmark(4), frame.size()),   // Nose tip
                relativeT(points.landmark(152), frame.size()), // Chin
                relativeT(points.landmark(263), frame.size()), // Left eye left corner
                relativeT(points.landmark(33), frame.size()),  // Right eye right corner
                relativeT(points.landmark(287), frame.size()), // Left Mouth corner
                relativeT(points.landmark(57), frame.size())   // Right mouth corner
            };

            cv::estimateAffine3D(image_points1, model_points, transformation, cv::noArray());

            cv::Mat pupil_world_cord;
            cv::Mat S;

            if (!transformation.empty())
            {
                // Project pupil image point into 3D world point
                cv::Mat pupil_world_cord_left = transformation * (cv::Mat_<double>(4, 1) << left_pupil.x, left_pupil.y, 0, 1);
                cv::Mat pupil_world_cord_right = transformation * (cv::Mat_<double>(4, 1) << right_pupil.x, right_pupil.y, 0, 1);

                // 3D gaze point
                cv::Mat S_left = (cv::Mat(Eye_ball_center_left) + (pupil_world_cord_left - cv::Mat(Eye_ball_center_left)) * 10);
                cv::Mat S_right = (cv::Mat(Eye_ball_center_right) + (pupil_world_cord_right - cv::Mat(Eye_ball_center_right)) * 10);

                pupil_world_cord = (pupil_world_cord_left + pupil_world_cord_right) * 0.5;
                S = (S_left + S_right) * 0.5;
            }
            else
            {
                if (yaw > 0)
                {
                    std::vector<cv::Point3d> image_points_left = {image_points1[0], image_points1[1], image_points1[2], image_points1[4]};
                    std::vector<cv::Point3d> model_points_left = {model_points[0], model_points[1], model_points[2], model_points[4]};
                    cv::estimateAffine3D(image_points_left, model_points_left, transformation, cv::noArray());
                    pupil_world_cord = transformation * (cv::Mat_<double>(4, 1) << left_pupil.x, left_pupil.y, 0, 1);
                    S = (cv::Mat(Eye_ball_center_left) + (pupil_world_cord - cv::Mat(Eye_ball_center_left)) * 10);
                }
                else
                {
                    std::vector<cv::Point3d> image_points_right = {image_points1[0], image_points1[1], image_points1[3], image_points1[5]};
                    std::vector<cv::Point3d> model_points_right = {model_points[0], model_points[1], model_points[3], model_points[5]};
                    cv::estimateAffine3D(image_points_right, model_points_right, transformation, cv::noArray());
                    pupil_world_cord = transformation * (cv::Mat_<double>(4, 1) << right_pupil.x, right_pupil.y, 0, 1);
                    S = (cv::Mat(Eye_ball_center_right) + (pupil_world_cord - cv::Mat(Eye_ball_center_right)) * 10);
                }
            }
            cv::Mat gaze_vector = S - pupil_world_cord;
            gaze_yaw = round(atan2(gaze_vector.at<double>(0, 0), gaze_vector.at<double>(2, 0)) * 180 / PI), 2;
            gaze_pitch = round(atan2(gaze_vector.at<double>(1, 0), gaze_vector.at<double>(2, 0)) * 180 / PI), 2;
        }

        std::pair<float, float> getGaze()
        {
            return std::make_pair(gaze_yaw, gaze_pitch);
        }
    }
}
