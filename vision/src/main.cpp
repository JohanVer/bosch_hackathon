#include "../include/main.h"

int main(int argc, char *argv[])
{
    std::cout << "Starting bosch vision node..." << std::endl;
    ros::init(argc, argv, "bosch_vision_node");
    ros::NodeHandle nh_;

    bosch_vision_class vision(nh_);
    vision.start();

    return 0;
}
