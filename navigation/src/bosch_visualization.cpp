#include "bosch_visualization.h"

namespace bosch_hackathon {

bosch_visualization::bosch_visualization(ros::NodeHandle &nh):
    nh_(nh)
{
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("bosch_marker", 1);
}

void bosch_visualization::sendCube(const std::string &frame, const std::string &ns, const int id, const tf::Vector3 &pos){
    uint32_t shape = visualization_msgs::Marker::CUBE;

    visualization_msgs::Marker marker;

    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time::now();

    marker.ns = ns;
    marker.id = id;

    marker.type = shape;

    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = pos.x();
    marker.pose.position.y = pos.y();
    marker.pose.position.z = pos.z();

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();

    marker_pub_.publish(marker);
}

}
