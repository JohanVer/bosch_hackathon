#ifndef BOSCH_NAVIGATION_H
#define BOSCH_NAVIGATION_H

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

#include "Eigen/Dense"
#include "bosch_visualization.h"
#include "occupancy_grid_utils/coordinate_conversions.h"
#include "occupancy_grid_utils/ray_tracer.h"
#include "tf_conversions/tf_eigen.h"

#define CIRCLE_DIVIDER 20.0
#define DIR_WEIGHT 1.0
#define RAY_WEIGHT 50.0
#define PLAN_RANGE 1.0 // Meters

namespace bosch_hackathon{

class bosch_navigation
{
public:
    bosch_navigation(ros::NodeHandle &nh);
    void start();
    void gridCallback(const nav_msgs::OccupancyGrid::ConstPtr& grid_msg);

private:
    float distToLine(const Eigen::Vector2f &v, const Eigen::Vector2f &w, const Eigen::Vector2f &p);
    float minimalDistToRay(const std::vector<geometry_msgs::Point> &occ_points, const Eigen::Vector2f &l1, const Eigen::Vector2f &l2 );
    std::vector<geometry_msgs::Point> listOccPoints();
    bool checkRayForCollision(const tf::Vector3 &tf_src, const tf::Vector3 &tf_dst);
    bool getTf(const std::string &target, const std::string &source, tf::StampedTransform &tf);
    double calcRayCost(const tf::StampedTransform &odom_T_baselink, const Eigen::Vector2f dst_e);
    double angleBetweenVectors(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2);
    bool evalPossibleDirections(tf::Vector3 &next_pos, double &new_heading);
    void sendGoal(const tf::Vector3 &position, const double &heading);

    ros::NodeHandle &nh_;
    ros::Publisher goal_publisher_;
    ros::Subscriber occ_map_sub_;
    tf::TransformListener tf_listener_;
    bosch_hackathon::bosch_visualization vis_;
    nav_msgs::OccupancyGrid current_local_map_;


    bool occ_initialized;
    uint32_t seq_id_;
};

}

#endif // BOSCH_NAVIGATION_H
