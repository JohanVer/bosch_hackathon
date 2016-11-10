#ifndef BOSCH_VISION_H
#define BOSCH_VISION_H

#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include "std_msgs/Header.h"
#include "nav_msgs/OccupancyGrid.h"
#include "geometry_msgs/PoseStamped.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <image_transport/image_transport.h>
#include <bosch_vision/BoschVision.h>

class colorFilter{
public:
    colorFilter(const std::string &window_name);

    cv::Mat filter(const cv::Mat &in);

private:
    int blur_kernel;
    int h_low;
    int h_high;
    int s_low;
    int s_high;
    int v_low;
    int v_high;
    int erode_iterations;
    int dilate_iterations;

    std::string window_name_;
};

class bosch_vision_class
{
public:
    bosch_vision_class(ros::NodeHandle &nh);
    void start();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    ros::NodeHandle &nh_;
    image_transport::ImageTransport it_;
    ros::Publisher vision_publisher_;
    image_transport::Subscriber img_sub_;

};

#endif // BOSCH_VISION_H
