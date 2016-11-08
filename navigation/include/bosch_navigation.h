#ifndef BOSCH_NAVIGATION_H
#define BOSCH_NAVIGATION_H

#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include "std_msgs/Header.h"
#include "nav_msgs/OccupancyGrid.h"

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

#include "Eigen/Dense"
#include "bosch_visualization.h"
#include "occupancy_grid_utils/coordinate_conversions.h"
#include "occupancy_grid_utils/ray_tracer.h"
#include "tf_conversions/tf_eigen.h"


namespace bosch_hackathon{

class bosch_navigation
{
public:
    bosch_navigation(ros::NodeHandle &nh);
    void start();
    void gridCallback(const nav_msgs::OccupancyGrid::ConstPtr& grid_msg);

private:
    ros::NodeHandle &nh_;
    ros::Publisher vel_cmd_publisher_;
    ros::Subscriber occ_map_sub_;
    tf::TransformListener tf_listener_;
    bosch_hackathon::bosch_visualization vis_;
    nav_msgs::OccupancyGrid current_local_map_;


    bool occ_initialized;
};

}

#endif // BOSCH_NAVIGATION_H
