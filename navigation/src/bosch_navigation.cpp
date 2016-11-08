#include "../include/bosch_navigation.h"

namespace bosch_hackathon{

bosch_navigation::bosch_navigation(ros::NodeHandle &nh):
    nh_(nh)
{
    occ_map_sub_ = nh_.subscribe("/map", 1, &bosch_navigation::gridCallback, this);
}

void bosch_navigation::start(){
    ros::Rate loop_rate(100);

    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }
}

void bosch_navigation::gridCallback(const nav_msgs::OccupancyGrid::ConstPtr& grid_msg)
{
    // Compute direction to go....

    // Get transformation to robot frame
    tf::StampedTransform transform;
    try{
        tf_listener_.lookupTransform("/odom", "/base_link",
                                 ros::Time(0), transform);
    }
    catch (tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
        ros::Duration(1.0).sleep();
    }

    std::cout << "Got transform with timestamp: " << transform.stamp_ << std::endl;

}


}
