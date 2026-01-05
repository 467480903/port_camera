#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>

class DualRTSPStream {
private:
    std::string rtsp_url1;
    std::string rtsp_url2;
    std::atomic<bool> running;
    
    cv::VideoCapture cap1;
    cv::VideoCapture cap2;
    
    std::mutex frame_mutex1;
    std::mutex frame_mutex2;
    cv::Mat current_frame1;
    cv::Mat current_frame2;
    
    std::thread capture_thread1;
    std::thread capture_thread2;
    
    int frame_width;
    int frame_height;
    double fps;

public:
    DualRTSPStream(const std::string& url1, const std::string& url2) 
        : rtsp_url1(url1), rtsp_url2(url2), running(false) {
    }

    ~DualRTSPStream() {
        stop();
    }

    bool initialize() {
        // Open first stream to get properties
        cap1.open(rtsp_url1, cv::CAP_FFMPEG);
        if (!cap1.isOpened()) {
            std::cerr << "Error: Could not open RTSP stream 1: " << rtsp_url1 << std::endl;
            return false;
        }

        // Get video properties from first stream
        frame_width = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_height = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = cap1.get(cv::CAP_PROP_FPS);
        
        if (fps <= 0) {
            fps = 30.0; // Default FPS if not available
        }

        std::cout << "Stream 1 properties: " << frame_width << "x" << frame_height 
                  << ", FPS: " << fps << std::endl;

        // Open second stream
        cap2.open(rtsp_url2, cv::CAP_FFMPEG);
        if (!cap2.isOpened()) {
            std::cerr << "Error: Could not open RTSP stream 2: " << rtsp_url2 << std::endl;
            cap1.release();
            return false;
        }

        std::cout << "Successfully opened both RTSP streams." << std::endl;
        return true;
    }

    void startCapture() {
        running = true;
        
        // Start capture threads
        capture_thread1 = std::thread(&DualRTSPStream::captureLoop1, this);
        capture_thread2 = std::thread(&DualRTSPStream::captureLoop2, this);
        
        std::cout << "Started capture threads." << std::endl;
    }

    void stop() {
        running = false;
        
        if (capture_thread1.joinable()) {
            capture_thread1.join();
        }
        if (capture_thread2.joinable()) {
            capture_thread2.join();
        }
        
        cap1.release();
        cap2.release();
        
        std::cout << "Stopped all capture threads." << std::endl;
    }

    cv::Mat getCombinedFrame() {
        std::lock_guard<std::mutex> lock1(frame_mutex1);
        std::lock_guard<std::mutex> lock2(frame_mutex2);
        
        // Check if we have both frames
        if (current_frame1.empty() || current_frame2.empty()) {
            // Return empty frame or a placeholder
            return cv::Mat();
        }
        
        // Resize frames if they have different dimensions
        cv::Mat frame1_resized, frame2_resized;
        cv::Size target_size(frame_width / 2, frame_height);
        
        cv::resize(current_frame1, frame1_resized, target_size);
        cv::resize(current_frame2, frame2_resized, target_size);
        
        // Create combined frame (side by side)
        cv::Mat combined_frame(frame_height, frame_width, current_frame1.type());
        
        // Copy first frame to left half
        cv::Rect left_roi(0, 0, frame_width / 2, frame_height);
        frame1_resized.copyTo(combined_frame(left_roi));
        
        // Copy second frame to right half
        cv::Rect right_roi(frame_width / 2, 0, frame_width / 2, frame_height);
        frame2_resized.copyTo(combined_frame(right_roi));
        
        // Add text labels
        cv::putText(combined_frame, "Camera 1", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(combined_frame, "Camera 2", cv::Point(frame_width / 2 + 10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Add dividing line
        cv::line(combined_frame, cv::Point(frame_width / 2, 0), 
                cv::Point(frame_width / 2, frame_height), cv::Scalar(255, 255, 255), 2);
        
        return combined_frame;
    }

private:
    void captureLoop1() {
        cv::Mat frame;
        while (running) {
            if (cap1.read(frame)) {
                std::lock_guard<std::mutex> lock(frame_mutex1);
                current_frame1 = frame.clone();
            } else {
                std::cerr << "Warning: Failed to read frame from stream 1. Reconnecting..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                
                // Try to reconnect
                cap1.release();
                cap1.open(rtsp_url1, cv::CAP_FFMPEG);
                if (!cap1.isOpened()) {
                    std::cerr << "Error: Failed to reconnect to stream 1." << std::endl;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void captureLoop2() {
        cv::Mat frame;
        while (running) {
            if (cap2.read(frame)) {
                std::lock_guard<std::mutex> lock(frame_mutex2);
                current_frame2 = frame.clone();
            } else {
                std::cerr << "Warning: Failed to read frame from stream 2. Reconnecting..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                
                // Try to reconnect
                cap2.release();
                cap2.open(rtsp_url2, cv::CAP_FFMPEG);
                if (!cap2.isOpened()) {
                    std::cerr << "Error: Failed to reconnect to stream 2." << std::endl;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

int main() {
    // RTSP URLs
    std::string rtsp_url1 = "rtsp://localhost:8554/cam3";
    std::string rtsp_url2 = "rtsp://localhost:8554/cam2";
    
    std::cout << "Starting Dual RTSP Stream Combiner..." << std::endl;
    std::cout << "Stream 1: " << rtsp_url1 << std::endl;
    std::cout << "Stream 2: " << rtsp_url2 << std::endl;
    
    // Create and initialize the dual stream combiner
    DualRTSPStream streamCombiner(rtsp_url1, rtsp_url2);
    
    if (!streamCombiner.initialize()) {
        std::cerr << "Failed to initialize RTSP streams. Exiting..." << std::endl;
        return -1;
    }
    
    // Start capturing
    streamCombiner.startCapture();
    
    // Create display window
    std::string windowName = "Combined RTSP Streams";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1280, 480);
    
    std::cout << "Press 'q' to quit, 's' to save current frame." << std::endl;
    
    int frame_count = 0;
    auto last_time = std::chrono::steady_clock::now();
    
    // Main display loop
    while (true) {
        // Get combined frame
        cv::Mat combined_frame = streamCombiner.getCombinedFrame();
        
        if (!combined_frame.empty()) {
            // Display the combined frame
            cv::imshow(windowName, combined_frame);
            
            // Calculate FPS
            frame_count++;
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time).count();
            
            if (elapsed >= 1) {
                std::cout << "Display FPS: " << frame_count / elapsed << std::endl;
                frame_count = 0;
                last_time = current_time;
            }
        } else {
            // Show waiting message
            cv::Mat waiting_frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::putText(waiting_frame, "Waiting for streams...", cv::Point(50, 240), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            cv::imshow(windowName, waiting_frame);
        }
        
        // Handle keyboard input
        char key = cv::waitKey(10) & 0xFF;
        
        if (key == 'q' || key == 'Q' || key == 27) {  // 'q' or ESC to quit
            std::cout << "Quitting..." << std::endl;
            break;
        } else if (key == 's' || key == 'S') {  // 's' to save frame
            if (!combined_frame.empty()) {
                std::string filename = "combined_frame_" + std::to_string(time(nullptr)) + ".jpg";
                cv::imwrite(filename, combined_frame);
                std::cout << "Saved frame to: " << filename << std::endl;
            }
        } else if (key == 'f' || key == 'F') {  // 'f' to toggle fullscreen
            static bool fullscreen = false;
            fullscreen = !fullscreen;
            if (fullscreen) {
                cv::setWindowProperty(windowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            } else {
                cv::setWindowProperty(windowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
            }
        }
    }
    
    // Cleanup
    cv::destroyAllWindows();
    streamCombiner.stop();
    
    std::cout << "Program terminated successfully." << std::endl;
    
    return 0;
}