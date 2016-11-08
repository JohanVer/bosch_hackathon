#include "../include/main.h"

int main(int argc, char *argv[])
{
    std::cout << "Starting bosch navigation node..." << std::endl;
    ros::init(argc, argv, "bosch_nav_node");
    ros::NodeHandle nh_;

    bosch_hackathon::bosch_navigation nav(nh_);
    nav.start();

    return 0;
}
