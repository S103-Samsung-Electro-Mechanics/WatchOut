// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>
#include <string>
#include <map>

#include <opencv2/opencv.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
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
#include "mediapipe/examples/desktop/run_graph_main.h"

#define LOG(msg) { std::cout << __func__ << " " << __LINE__ << " " << msg << std::endl; }

constexpr char kInputStream[] = "input_video";
constexpr char kWindowName[] = "MediaPipe";

constexpr char kVideoOutputStream[] = "output_video";
constexpr char kLandmarksOutputStream[] = "face_landmarks_with_iris";
constexpr char kLandmarkPresenceOutputStream[] = "landmark_presence";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

static inline absl::Status createGraphFromFile(
  std::string calculator_graph_config_file,
  mediapipe::CalculatorGraph& graph) {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
    calculator_graph_config_file,
    &calculator_graph_config_contents));
  
  mediapipe::CalculatorGraphConfig config =
    mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
      calculator_graph_config_contents);
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  return absl::OkStatus();
}

class MPPGraphRunner {
  private:
  mediapipe::CalculatorGraph graph;
  mediapipe::GlCalculatorHelper gpu_helper;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller_video;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmark_presence;

  public:
  absl::Status initMPPGraph(std::string calculator_graph_config_file) {
    MP_RETURN_IF_ERROR(createGraphFromFile(calculator_graph_config_file, graph));

    MP_ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
    MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
    gpu_helper.InitializeForTest(graph.GetGpuResources().get());

    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_video_,
      graph.AddOutputStreamPoller(kVideoOutputStream));
    this->poller_video = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_video_));

    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_,
      graph.AddOutputStreamPoller(kLandmarksOutputStream));
    this->poller_landmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_));

    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark_presence_,
      graph.AddOutputStreamPoller(kLandmarkPresenceOutputStream));
    this->poller_landmark_presence = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmark_presence_));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    return absl::OkStatus();
  }

  absl::Status processFrame(
    cv::Mat& camera_frame,
    size_t frame_timestamp_us,
    cv::Mat& output_frame_mat,
    ::mediapipe::NormalizedLandmarkList& landmarks,
    bool& landmark_presence
  ) {
    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
      mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);    
    MP_RETURN_IF_ERROR(
      this->gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, this]() -> absl::Status {
        // Convert ImageFrame to GpuBuffer.
        auto texture = this->gpu_helper.CreateSourceTexture(*input_frame.get());
        auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
        glFlush();
        texture.Release();

        // Send GPU image packet into the graph.
        auto status = this->graph.AddPacketToInputStream(
          kInputStream,
          mediapipe::Adopt(gpu_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us)));
        // ABSL_LOG(INFO) << status;
        
        return absl::OkStatus();
      }));
    
    // Get the graph result packet, or stop if that fails
    mediapipe::Packet packet_video, packet_landmarks, packet_landmark_presence;
    this->poller_video->Next(&packet_video);
    if (this->poller_landmark_presence->QueueSize() > 0) {
      this->poller_landmark_presence->Next(&packet_landmark_presence);
      landmark_presence = packet_landmark_presence.Get<bool>();
      if (landmark_presence) {
        this->poller_landmarks->Next(&packet_landmarks);
        landmarks = packet_landmarks.Get<::mediapipe::NormalizedLandmarkList>();
      }
    }
	
    // Convert GpuBuffer to ImageFrame.
    std::unique_ptr<mediapipe::ImageFrame> output_frame;
    MP_RETURN_IF_ERROR(
      this->gpu_helper.RunInGlContext([&packet_video, &output_frame, this]() -> absl::Status {
        auto &gpu_frame = packet_video.Get<mediapipe::GpuBuffer>();
        auto texture = this->gpu_helper.CreateSourceTexture(gpu_frame);
        output_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
          gpu_frame.width(), gpu_frame.height(),
          mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        this->gpu_helper.BindFramebuffer(texture);
        const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
          gpu_frame.format(), 0, this->gpu_helper.GetGlVersion());
        glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format, info.gl_type, output_frame->MutablePixelData());
        glFlush();
        texture.Release();
        return absl::OkStatus();
      }));
    // Convert back to opencv for display or saving.
    output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    if (output_frame_mat.channels() == 4)
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
    else
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    
    return absl::OkStatus();
  }
};

bool MPPGraphRunnerWrapper::initMPPGraph(std::string calculator_graph_config_file) {
  this->core_runner_ptr = static_cast<void*>(new MPPGraphRunner());
  MPPGraphRunner& runner = *(static_cast<MPPGraphRunner*>(this->core_runner_ptr));

  absl::Status status = runner.initMPPGraph(calculator_graph_config_file);
  if (!status.ok())
    std::cerr << "Failed to initialize the graph." << status.message() << std::endl;
  
  return status.ok();
}
bool MPPGraphRunnerWrapper::processFrame(
  cv::Mat& camera_frame,
  size_t frame_timestamp_us,
  cv::Mat& output_frame_mat,
  DMSLandmarks& dms_landmarks,
  bool& landmark_presence) {
  MPPGraphRunner& runner = *(static_cast<MPPGraphRunner*>(this->core_runner_ptr));
  ::mediapipe::NormalizedLandmarkList landmarks_;
  absl::Status status = runner.processFrame(camera_frame, frame_timestamp_us, output_frame_mat, landmarks_, landmark_presence);
  if (!status.ok()) {
    std::cerr << "Failed to process the frame." << status.message() << std::endl;
    return status.ok();
  }

  if (landmark_presence) {
    for (int i = 0; i < 18; ++i) {
      dms_landmarks.landmarks[i].x = landmarks_.landmark(landmark_converting_table[i]).x();
      dms_landmarks.landmarks[i].y = landmarks_.landmark(landmark_converting_table[i]).y();
      dms_landmarks.landmarks[i].z = landmarks_.landmark(landmark_converting_table[i]).z();
    }
  }
  return status.ok();
}
MPPGraphRunnerWrapper::~MPPGraphRunnerWrapper() {
  delete static_cast<MPPGraphRunner*>(this->core_runner_ptr);
}
