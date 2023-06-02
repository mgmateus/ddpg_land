#include <ros/package.h>
#include <ros/ros.h>
#include "std_msgs/Float64MultiArray.h"

#include "../include/vision.hpp"
#include <active_perception_pkgs/ProcessedImg.h>

Vision vision;

ros::Publisher processedMsg_pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    rl_pkg::ProcessedImg processedMsg;
    
    try{
        vision.setImage(cv_bridge::toCvShare(msg, "bgr8")->image);
        vision.processImage();
        processedMsg.dxPixel = vision.dxPixel;
        processedMsg.dyPixel = vision.dyPixel;
        
        processedMsg_pub.publish(processedMsg);
    }catch (cv_bridge::Exception& e){
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
    
}


int main(int argc, char *argv[]){
    ros::init(argc, argv, "cam_listener");

    
    ros::NodeHandle node;
    
    processedMsg_pub = node.advertise<rl_pkg::ProcessedImg>("/vision", 2);
    

    image_transport::ImageTransport it(node);
	image_transport::Subscriber sub = it.subscribe("/airsim_node/drone_1/front_center_custom/Scene", 1, imageCallback);
   
    ros::spin();
}
