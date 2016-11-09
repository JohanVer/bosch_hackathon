#include "../include/bosch_navigation.h"

namespace bosch_hackathon{

bosch_navigation::bosch_navigation(ros::NodeHandle &nh):
    nh_(nh),
    vis_(nh),
    occ_initialized(false),
    seq_id_(0)
{
    occ_map_sub_ = nh_.subscribe("/move_base/local_costmap/costmap", 1, &bosch_navigation::gridCallback, this);
    goal_publisher_ = nh_.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1);

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

void bosch_navigation::sendGoal(const tf::Vector3 &position, const double &heading){
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "/odom";
    pose.header.stamp = ros::Time::now();
    pose.header.seq = seq_id_;
    seq_id_++;

    pose.pose.position.x = position.x();
    pose.pose.position.y = position.y();
    pose.pose.position.z = position.z();

    tf::Quaternion rot;
    rot.setEulerZYX(heading, 0,0);
    pose.pose.orientation.x = rot.x();
    pose.pose.orientation.y = rot.y();
    pose.pose.orientation.z = rot.z();
    pose.pose.orientation.w = rot.w();

    goal_publisher_.publish(pose);
}

float bosch_navigation::distToLine(const Eigen::Vector2f &v, const Eigen::Vector2f &w, const Eigen::Vector2f &p){
    // Return minimum distance between line segment vw and point p
    const float l2 = (w-v).norm();
    if (l2 == 0.0) return (p-v).norm();   // v == w case
    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line.
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    // We clamp t from [0,1] to handle points outside the segment vw.
    float proj_len = ((p-v).dot(w-v) / l2);

    const float t = std::min((float)1.0, proj_len);
    if(t > 0){
        const Eigen::Vector2f projection = v + t * (w - v);  // Projection falls on the segment
        return (p-projection).norm();
    }
    return -1;
}

float bosch_navigation::minimalDistToRay(const std::vector<geometry_msgs::Point> &occ_points, const Eigen::Vector2f &l1, const Eigen::Vector2f &l2 ){
    float min_dist = std::numeric_limits<float>::max();
    bool point_perp = false;
    for(auto i = 0; i < occ_points.size(); i++){
        const geometry_msgs::Point &p = occ_points.at(i);
        Eigen::Vector2f p_e(p.x, p.y);
        float ray_to_p = distToLine(l1, l2, p_e);
        if(ray_to_p != -1){
            point_perp = true;
            // Update minimal distance
            if(ray_to_p < min_dist) min_dist = ray_to_p;
            tf::Vector3 p_tf = geoToTf(p);
            vis_.sendCube("/odom", "goals",i+2, p_tf);
        }
    }

    if(point_perp){
        return min_dist;
    }else{
        return 0;
    }
}

std::vector<geometry_msgs::Point> bosch_navigation::listOccPoints(){
    // Find all occupied cells within map
    std::vector<geometry_msgs::Point> occ_points;
    int occ_cells = 0;
    for(auto i = 0; i < current_local_map_.data.size(); i++){
        uint32_t index = current_local_map_.data.at(i);
        if(index == occupancy_grid_utils::OCCUPIED){
            occupancy_grid_utils::Cell cell = occupancy_grid_utils::indexCell(current_local_map_.info, i);
            occ_points.push_back(occupancy_grid_utils::cellCenter(current_local_map_.info, cell));
            occ_cells++;
        }
    }
    std::cout << "Occopied cells: " << occ_cells << " , total size is: " << current_local_map_.data.size() << std::endl;
    return occ_points;
}

bool bosch_navigation::checkRayForCollision(const tf::Vector3 &tf_src, const tf::Vector3 &tf_dst){
    occupancy_grid_utils::RayTraceIterRange ray = occupancy_grid_utils::rayTrace(current_local_map_.info, tfToGeo(tf_src), tfToGeo(tf_dst));
    int ray_cost = 0;
    for(auto i = ray.first; i != ray.second; i++){
        uint32_t index = occupancy_grid_utils::cellIndex(current_local_map_.info, *i);
        if(current_local_map_.data[index] >= 90){
            std::cout << "Collision detected... "<< std::endl;
            return true;
        }
        ray_cost += current_local_map_.data[index];
    }
    //std::cout << "Cost of ray is: " << ray_cost << std::endl;
    return false;
}

double bosch_navigation::angleBetweenVectors(const Eigen::Vector2f &vec1, const Eigen::Vector2f &vec2){
    double dot = vec1.dot(vec2);
    double det = vec1[0]*vec2[1] - vec1[1]*vec2[0];
    return std::atan2(det, dot);
}

bool bosch_navigation::getTf(const std::string &target, const std::string &source, tf::StampedTransform &tf){
    // Get transformation to robot frame
    tf::StampedTransform transform;
    try{
        tf_listener_.lookupTransform(target,source,
                                     ros::Time(0), transform);
    }
    catch (tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
        return false;
    }
    //std::cout << "Got transform with timestamp: " << transform.stamp_ << std::endl;
    tf = transform;
    return true;
}

double bosch_navigation::calcRayCost(const tf::StampedTransform &odom_T_baselink, const Eigen::Vector2f dst_e){
    tf::Vector3 dst(dst_e[0],dst_e[1],0);
    tf::Vector3 src(0,0,0);
    tf::Vector3 tf_dst = odom_T_baselink * dst;
    tf::Vector3 tf_src = odom_T_baselink * src;

    bool collision = checkRayForCollision(tf_src, tf_dst);
    if(collision) return std::numeric_limits<double>::max();

    std::vector<geometry_msgs::Point> occ_points = listOccPoints();

    // Compute perpendicular distance between ray spanned by l1 and l2 and point P
    Eigen::Vector2f l1(tf_src.x(), tf_src.y());
    Eigen::Vector2f l2(tf_dst.x(), tf_dst.y());
    float cost_dist_ray = 1.0 / minimalDistToRay(occ_points, l1, l2);
    std::cout << "Cost dist ray: " << cost_dist_ray << std::endl;
    return cost_dist_ray;
}

bool bosch_navigation::evalPossibleDirections(tf::Vector3 &next_pos, double &new_heading){
    tf::StampedTransform transform;
    if(!getTf("/odom", "/base_link", transform)){
        return false;
    }

    double roll, pitch, yaw;
    tf::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
    std::cout << "Actual yaw angle: " << yaw * 180.0/M_PI << std::endl;

    double increment = (2 * M_PI) / CIRCLE_DIVIDER;

    const Eigen::Vector2f front_dir(1.0 , 0.0);
    const tf::Vector3 front_dir_tf(PLAN_RANGE , 0.0, 0.0);
    Eigen::Vector2f best_p;
    double min_c = std::numeric_limits<double>::max();
    double angle_save = 0;
    for (double r = 0.0; r < 2 * M_PI; r += increment ){
        tf::Transform rotate;
        tf::Quaternion rot_quat;
        rot_quat.setEulerZYX(r,0,0);
        rotate.setRotation(rot_quat);
        rotate.setOrigin(tf::Vector3(0,0,0));

        tf::Vector3 test_p = rotate * front_dir_tf;


        Eigen::Vector2f eval_dir(test_p.x() , test_p.y());

        double angle = angleBetweenVectors(front_dir, eval_dir);
        double cost_direction = DIR_WEIGHT * std::fabs(angle) * 180.0/M_PI;
        double ray_cost = RAY_WEIGHT * calcRayCost(transform, eval_dir);

        double total_cost = cost_direction + ray_cost;

        if(total_cost < min_c){
            min_c = total_cost;
            best_p = eval_dir;
            angle_save = angle;
        }

        std::cout << "Cost direction: " << cost_direction << " , Cost ray: " << ray_cost << " total cost: " << total_cost << std::endl;
    }

    std::cout << "Best cost: " << min_c << ", Best direction: " << best_p << std::endl;

    // Transform point into odom
    tf::Vector3 dst(best_p[0],best_p[1],0);
    tf::Vector3 src(0,0,0);
    tf::Vector3 tf_dst = transform * dst;
    tf::Vector3 tf_src = transform * src;
    // Visualize direction
    vis_.sendCube("/odom", "goals",0, tf_dst);
    vis_.sendCube("/odom", "goals",1, tf_src);

    // These values are in odom
    new_heading = yaw + angle_save;
    next_pos = tf_dst;

    return true;
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

        tf::Vector3 next_dir;
        double new_heading = 0;
        if(evalPossibleDirections(next_dir, new_heading)){
            sendGoal(next_dir, new_heading);
        }
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
