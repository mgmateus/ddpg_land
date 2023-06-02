#pragma once

#include <image_transport/image_transport.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/viz/types.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float64.h>
#include "std_msgs/Float64MultiArray.h"
#include <math.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

using namespace cv;
using namespace std;



class Vision{
    public:
        
        Mat imgBGR2HSV, imgBGR2GRAY_t_1, imgBGR2GRAY, imgLight, imgThresh, imgCanny;
        Mat imgScene_t_1, imgScene, imgMain;

        bool firstView;
        Mat morph_kernel;

        vector<vector<Point>> hullLight, hullBoat;
        vector<Point> ContourBoat;
        Rect rectBoat;
        Point pointBoat;



        vector<vector<Point>> findContour(Mat img);
        void findLight();
        void findBoat();

    
        Vision();
        void setImage(Mat imgCam);
        void processImage();

        float dxPixel;
        float dyPixel;
        
};