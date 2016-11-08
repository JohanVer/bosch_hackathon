#include "../include/bosch_navigation.h"

namespace bosch_hackathon{

bosch_navigation::bosch_navigation(ros::NodeHandle &nh):
    nh_(nh),
    vis_(nh),
    occ_initialized(false)
{
    occ_map_sub_ = nh_.subscribe("/move_base/local_costmap/costmap", 1, &bosch_navigation::gridCallback, this);

}

geometry_msgs::Point tfToGeo(const tf::Vector3 &vec){
    geometry_msgs::Point vec_geo;
    vec_geo.x = vec.x();
    vec_geo.y = vec.y();
    vec_geo.z = vec.z();
    return vec_geo;
}

tf::Vector3 geoToTf(const geometry_msgs::Point &vec){
    tf::Vector3 vec_tf;
    vec_tf.setX(vec.x);
    vec_tf.setY(vec.y);
    vec_tf.setZ(vec.z);
    return vec_tf;
}

float distToLine(const Eigen::Vector2f &l1, const Eigen::Vector2f &l2, const Eigen::Vector2f &p){
   return (((l2-l1).dot(l1-p))).norm()/(l2-l1).norm();
}

void bosch_navigation::start(){
    ros::Rate loop_rate(3);

    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();

        if(!occ_initialized){
            std::cout << "Waiting for occ-map" << std::endl;
            continue;
        }

        // Get transformation to robot frame
        tf::StampedTransform transform;
        try{
            tf_listener_.lookupTransform("/odom","/base_link",
                                         ros::Time(0), transform);
        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            return;
        }
        std::cout << "Got transform with timestamp: " << transform.stamp_ << std::endl;

        tf::Vector3 dst(1,0,0);
        tf::Vector3 src(0,0,0);
        tf::Vector3 tf_dst = transform * dst;
        tf::Vector3 tf_src = transform * src;
        vis_.sendCube("/odom", "goals",0, tf_dst);
        vis_.sendCube("/odom", "goals",1, tf_src);

        occupancy_grid_utils::RayTraceIterRange ray = occupancy_grid_utils::rayTrace(current_local_map_.info, tfToGeo(tf_src), tfToGeo(tf_dst));

        int ray_cost = 0;
        for(auto i = ray.first; i != ray.second; i++){
            uint32_t index = occupancy_grid_utils::cellIndex(current_local_map_.info, *i);
            ray_cost += current_local_map_.data[index];
        }
        std::cout << "Cost of ray is: " << ray_cost << std::endl;

        int occ_cells = 0;
        std::vector<geometry_msgs::Point> occ_points;
        for(auto i = 0; i < current_local_map_.data.size(); i++){
            uint32_t index = current_local_map_.data.at(i);
            if(index == occupancy_grid_utils::OCCUPIED){
                occupancy_grid_utils::Cell cell = occupancy_grid_utils::indexCell(current_local_map_.info, i);
                occ_points.push_back(occupancy_grid_utils::cellCenter(current_local_map_.info, cell));
                occ_cells++;
            }
        }

        // Compute ray in parametric line form


        for(auto i = 0; i < occ_points.size(); i++){
            tf::Vector3 p = geoToTf(occ_points.at(i));
            vis_.sendCube("/odom", "goals",i+2, p);
        }

        std::cout << "Occopied cells: " << occ_cells << " , total size is: " << current_local_map_.data.size() << std::endl;
    }
}

void bosch_navigation::gridCallback(const nav_msgs::OccupancyGrid::ConstPtr& grid_msg)
{
    // Compute direction to go....
    current_local_map_ = *grid_msg;
    occ_initialized = true;

    std::cout << "Got grid message..." << grid_msg->header.stamp << std::endl;

}


}
