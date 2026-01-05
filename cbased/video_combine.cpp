#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <output_video> <input_video1> <input_video2> [input_video3 ...] " << std::endl;
        return -1;
    }

    std::string outputVideo = argv[1];
    std::vector<std::string> inputVideos;
    for(int i = 2; i < argc; ++i) {
        inputVideos.push_back(argv[i]);
    }

    cv::VideoCapture firstVideo(inputVideos[0]);
    if (!firstVideo.isOpened()) {
        std::cerr << "Error: Could not open video file " << inputVideos[0] << std::endl;
        return -1;
    }

    int ex = static_cast<int>(firstVideo.get(cv::CAP_PROP_FOURCC));
    Size S = Size((int) firstVideo.get(cv::CAP_PROP_FRAME_WIDTH), (int) firstVideo.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = firstVideo.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer;
    writer.open(outputVideo, ex, fps, S, true);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not create video file " << outputVideo << std::endl;
        return -1;
    }

    cv::Mat frame;
    for (const auto& videoPath : inputVideos) {
        cv::VideoCapture video(videoPath);
        if (!video.isOpened()) {
            std::cerr << "Error: Could not open video file " << videoPath << std::endl;
            continue;
        }

        while (video.read(frame)) {
            writer.write(frame);
        }

        video.release();
    }

    writer.release();
    firstVideo.release();
    std::cout << "Video combine complete. Output file: " << outputVideo << std::endl;

    return 0;
}