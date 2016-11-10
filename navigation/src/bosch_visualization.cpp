#include "bosch_visualization.h"

namespace bosch_hackathon {

double bosch_visualization::angleBetweenVectors(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2){
    double dot = vec1.dot(vec2);
    double det = vec1[0]*vec2[1] - vec1[1]*vec2[0];
    return std::atan2(det, dot);
}

bosch_visualization::bosch_visualization(ros::NodeHandle &nh):
    nh_(nh)
{
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("bosch_marker", 1);
    marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("bosch_rays",1);
}

visualization_msgs::Marker bosch_visualization::createMarker(const std::string &frame, const std::string &ns, const int id, const tf::Vector3 &pos, const tf::Quaternion rot, tf::Vector3 &scale, uint32_t shape, std_msgs::ColorRGBA color){
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

    marker.pose.orientation.x = rot.x();
    marker.pose.orientation.y = rot.y();
    marker.pose.orientation.z = rot.z();
    marker.pose.orientation.w = rot.w();

    marker.scale.x = scale.x();
    marker.scale.y = scale.y();
    marker.scale.z = scale.z();

    marker.color.r = color.r;
    marker.color.g = color.g;
    marker.color.b = color.b;
    marker.color.a = color.a;

    marker.lifetime = ros::Duration();
    return marker;
}

void bosch_visualization::sendCube(const std::string &frame, const std::string &ns, const int id, const tf::Vector3 &pos){
    uint32_t shape = visualization_msgs::Marker::CUBE;

    tf::Vector3 scale(0.1, 0.1, 0.1);
    tf::Quaternion rot(0,0,0,1);
    std_msgs::ColorRGBA color;
    color.r = 0.0f;
    color.g = 1.0f;
    color.b = 0.0f;
    color.a = 1.0;

    visualization_msgs::Marker marker = createMarker(frame, ns, id, pos, rot, scale, shape, color);

    marker_pub_.publish(marker);
}

void bosch_visualization::sendRays(const std::string &frame, const std::string &ns, const int id,const tf::Vector3 &origin, const std::vector<std::pair<tf::Vector3, enum RAY_COLOR>> &rays_endpoints){
    visualization_msgs::MarkerArray rays;
    uint32_t shape = visualization_msgs::Marker::ARROW;

    for(auto i = 0; i < rays_endpoints.size(); i++){
        tf::Quaternion quat;
        const tf::Vector3 &e_p = rays_endpoints.at(i).first;
        double angle = angleBetweenVectors(Eigen::Vector2f(origin.x(),origin.y()), Eigen::Vector2f(e_p.x(), e_p.y()));
        quat.setEulerZYX(angle,0,0);

        tf::Vector3 scale(1, 0.03, 0.03);
        std_msgs::ColorRGBA color;

        if(rays_endpoints.at(i).second == RAY_COLOR::FREE){
            color.r = 0.0f;
            color.g = 0.0f;
            color.b = 0.0f;
            color.a = 1.0;
        }else if(rays_endpoints.at(i).second == RAY_COLOR::USED){
            color.r = 0.0f;
            color.g = 1.0f;
            color.b = 0.0f;
            color.a = 1.0;
        }else if(rays_endpoints.at(i).second == RAY_COLOR::BLOCKED){
            color.r = 1.0f;
            color.g = 0.0f;
            color.b = 0.0f;
            color.a = 1.0;
        }

        visualization_msgs::Marker marker = createMarker(frame, ns, i, tf::Vector3(0,0,0), quat, scale, shape, color);
        rays.markers.push_back(marker);

    }
    marker_array_pub_.publish(rays);

}

}
