#ifndef BOSCH_VISUALIZATION_H
#define BOSCH_VISUALIZATION_H

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "ros/ros.h"
#include "tf/tf.h"
#include "Eigen/Dense"

namespace bosch_hackathon{

enum RAY_COLOR{
    FREE,
    USED,
    BLOCKED
};

class bosch_visualization
{
public:
    bosch_visualization(ros::NodeHandle &nh);
    void sendCube(const std::string &frame, const std::string &ns, const int id, const tf::Vector3 &pos);
    visualization_msgs::Marker createMarker(const std::string &frame, const std::string &ns, const int id, const tf::Vector3 &pos, const tf::Quaternion rot, tf::Vector3 &scale, uint32_t shape, std_msgs::ColorRGBA color);
    double angleBetweenVectors(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2);
    void sendRays(const std::string &frame, const std::string &ns, const int id, const tf::Vector3 &origin, const std::vector<std::pair<tf::Vector3, RAY_COLOR> > &rays_endpoints);

private:
    ros::NodeHandle &nh_;
    ros::Publisher marker_pub_;
    ros::Publisher marker_array_pub_;

};

}

#endif // BOSCH_VISUALIZATION_H
