#include "../include/bosch_vision.h"

colorFilter::colorFilter(const std::string &window_name){
    blur_kernel = 5;
    h_low = 0;
    h_high = 180;
    s_low = 0;
    s_high = 255;
    v_low = 0;
    v_high = 255;
    erode_iterations = 0;
    dilate_iterations = 0;
    window_name_ = window_name;

    cv::namedWindow(window_name_, 1);
    cv::createTrackbar("Blur-Kernel", window_name_, &blur_kernel, 20);
    cv::createTrackbar("h_low", window_name_, &h_low, 180);
    cv::createTrackbar("h_high", window_name_, &h_high, 180);
    cv::createTrackbar("s_low", window_name_, &s_low, 255);
    cv::createTrackbar("s_high", window_name_, &s_high, 255);
    cv::createTrackbar("v_low", window_name_, &v_low, 255);
    cv::createTrackbar("v_high", window_name_, &v_high, 255);
    cv::createTrackbar("erode", window_name_, &erode_iterations, 20);
    cv::createTrackbar("dilate", window_name_, &dilate_iterations, 20);


}

cv::Mat colorFilter::filter(const cv::Mat &in){
    cv::Mat img;
    cv::blur( in, img, cv::Size( blur_kernel, blur_kernel));

    cv::Mat img_hsv;
    cv::cvtColor(img,img_hsv,CV_BGR2HSV);

    cv::Mat thres_img;
    cv::inRange(img_hsv, cv::Scalar(h_low, s_low, v_low),cv::Scalar(h_high, s_high, v_high), thres_img);

    cv::erode(thres_img, thres_img, cv::Mat(), cv::Point(-1, -1), erode_iterations);
    cv::dilate(thres_img, thres_img, cv::Mat(), cv::Point(-1, -1), dilate_iterations);

    cv::imshow(window_name_, thres_img);
    return thres_img;
}

bosch_vision_class::bosch_vision_class(ros::NodeHandle &nh):
    nh_(nh),
    it_(nh)
{
    vision_publisher_ = nh_.advertise<bosch_vision::BoschVision>("/move_base_simple/goal", 1);
    img_sub_ = it_.subscribe("/Camera1/image_rect_color", 2, &bosch_vision_class::imageCallback, this);
}

void bosch_vision_class::imageCallback(const sensor_msgs::ImageConstPtr& msg){

}

void bosch_vision_class::start(){
    ros::Rate loop_rate(10);

    colorFilter filter1("red");


    while (ros::ok()) {
        cv::Mat img = cv::imread("/home/vertensj/Desktop/red_obst.png",1);

        ros::spinOnce();
        loop_rate.sleep();

        std::cout << "Process image... " << std::endl;

        cv::Mat thres_img = filter1.filter(img);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(thres_img.clone(), contours, CV_RETR_TREE,
                         CV_CHAIN_APPROX_NONE);

        for (int i = 0; i < contours.size(); i++) {
            std::vector<cv::Point> approx;
            std::vector<cv::Point> &act_contour = contours.at(i);
            cv::approxPolyDP(cv::Mat(act_contour), approx,
                             cv::arcLength(cv::Mat(act_contour), true) * 0.04, true);

            double size = cv::contourArea(act_contour);
            std::cout << "Contour: " << size << " approx: " << approx.size() << std::endl;

            if(size > 1000){
                if (approx.size() == 4) {
                    std::cout << "Rectangle..." << std::endl;
                    cv::drawContours(img, contours, i, cv::Scalar(0, 255, 255), CV_FILLED);
                }else if(approx.size() == 3){
                    std::cout << "Triangle..." << std::endl;
                    cv::drawContours(img, contours, i, cv::Scalar(0, 255, 0), CV_FILLED);
                }else{
                    cv::drawContours(img, contours, i, cv::Scalar(255, 0, 0), CV_FILLED);
                }

                //Calculate mass center
                cv::Moments mu = cv::moments(contours[i], false);
                cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
                cv::circle(img, mc, 5, cv::Scalar(0, 0, 255));
            }
        }

        cv::imshow("Original smoothed", img);
        int iKey = cv::waitKey(50);
        if (iKey == 27)
        {
            break;
        }

    }
}
