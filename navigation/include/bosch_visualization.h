#ifndef BOSCH_VISUALIZATION_H
#define BOSCH_VISUALIZATION_H

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "ros/ros.h"
#include "tf/tf.h"

namespace bosch_hackathon{

class bosch_visualization
{
public:
    bosch_visualization(ros::NodeHandle &nh);
    void sendCube(const std::string &frame, const std::string &ns, const int id, const tf::Vector3 &pos);

private:
    ros::NodeHandle &nh_;
    ros::Publisher marker_pub_;

};

}

#endif // BOSCH_VISUALIZATION_H
