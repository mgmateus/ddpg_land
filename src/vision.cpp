#include <ros/package.h>
#include <ros/ros.h>
#include "std_msgs/Float64MultiArray.h"

#include "../include/vision.hpp"

Vision::Vision(){
    this->morph_kernel = Mat::ones(9, 9, CV_8U);
    this->firstView = true;
}

void Vision::setImage(Mat imgCam){
    if (this->firstView){
        imgCam.copyTo(this->imgScene_t_1);
        this->imgScene = Mat::ones(imgCam.rows, imgCam.cols, CV_8U);
        this->imgMain = Mat::ones(imgCam.rows, imgCam.cols, CV_8U);
        
    }
    else{
        imgCam.copyTo(this->imgScene);
        imgCam.copyTo(this->imgMain);
    }
    
}

vector<vector<Point>> Vision::findContour(Mat img){
    Mat processed_image = Mat::ones(img.rows, img.cols, CV_8U);
    vector<vector<Point>> contours;
    
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> hull( contours.size() );

    for( size_t i = 0; i < contours.size(); i++ ) {
        convexHull( contours[i], hull[i] );
    }

    return hull;
}   

void Vision::findLight(){
    try{

        if(this->firstView){
            cvtColor(this->imgScene_t_1, this->imgBGR2GRAY_t_1, COLOR_BGR2GRAY);
            GaussianBlur(this->imgBGR2GRAY_t_1, this->imgBGR2GRAY_t_1, Size(3,3), 0);
            threshold(this->imgBGR2GRAY_t_1, this->imgThresh, 0, 255, THRESH_BINARY+THRESH_OTSU);

            this->imgLight = Mat::ones(this->imgBGR2GRAY_t_1.rows, this->imgBGR2GRAY_t_1.cols, CV_8U);
            this->imgBGR2GRAY = Mat::ones(this->imgBGR2GRAY_t_1.rows, this->imgBGR2GRAY_t_1.cols, CV_8U);
            this->imgCanny = Mat::ones(this->imgBGR2GRAY_t_1.rows, this->imgBGR2GRAY_t_1.cols, CV_8U);
            
        }
        else{
            this->imgLight = Mat::ones(this->imgBGR2GRAY_t_1.rows, this->imgBGR2GRAY_t_1.cols, CV_8U);
            cvtColor(this->imgScene, this->imgBGR2GRAY, COLOR_BGR2GRAY);
            GaussianBlur(this->imgBGR2GRAY, this->imgBGR2GRAY, Size(3,3), 0);
            threshold(this->imgBGR2GRAY, this->imgThresh, 0, 255, THRESH_BINARY+THRESH_OTSU); 
            
            morphologyEx(this->imgThresh, this->imgThresh, MORPH_CLOSE, this->morph_kernel, Point(-1,-1), 4);
            
            Mat mask = Mat::zeros(this->imgBGR2GRAY.size(), this->imgBGR2GRAY.type());
            vector<Point2f> p0, p1, light;
            
            goodFeaturesToTrack(this->imgBGR2GRAY_t_1, p0, 1000, 0.1, 7, Mat(), 7, false, 0.04);
            vector<uchar> status;
            vector<float> err;
            TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01);
            calcOpticalFlowPyrLK(this->imgBGR2GRAY_t_1, this->imgBGR2GRAY, p0, p1, status, err, Size(21,21), 1, criteria);

         
            vector<Point> l(p0.size());
            vector<Point2f> good_new;
            for(uint i = 0; i < p0.size(); i++)
            {
                // Select good points
                if(status[i] == 1) {
                    
                    good_new.push_back(p1[i]);
                    // draw the tracks
                    float d = hypot(p1[i].x-p0[i].x, p1[i].y-p0[i].y);
                    if(d >5 && d <= 50){
                        light.push_back(p1[i]);
                        line(this->imgMain,p1[i], p0[i], viz::Color::yellow(), 2);
                        circle(this->imgMain, p1[i], 5, viz::Color::red(), -1);

                        l[i].x = p1[i].x;
                        l[i].y = p1[i].y;

                    }
                    
                    
                    
                }
            }
            Point pointLight;
            double x = 0;
            double y = 0;
            for (unsigned int i=0; i < light.size(); i++){
                x += light[i].x;
                y += light[i].y;
            }

            x /= light.size();
            y /= light.size();

            pointLight.x = x;
            pointLight.y = y;

            Rect rectLigth;
            if(pointLight.y <= this->imgMain.rows/2){
                rectLigth.x = pointLight.x - 100;
                rectLigth.y = 0;
            }
            else{
                rectLigth.x = pointLight.x - 20;
                rectLigth.y = pointLight.y - 20;
            }

            rectLigth.width = 200;
            rectLigth.height = 150;
            
            rectangle(this->imgLight, rectLigth, viz::Color::white(), -1);
            circle(this->imgMain, pointLight, 7, viz::Color::purple(), -1);
            add(this->imgMain, mask, this->imgMain);
            
            this->imgBGR2GRAY.copyTo(this->imgBGR2GRAY_t_1);
            

        }
    }
    catch(Exception& e){
        ROS_WARN("...");
        
    }
    
}

void Vision::findBoat(){

    try{

        subtract(this->imgBGR2GRAY, this->imgLight, this->imgBGR2GRAY);
        Canny(this->imgBGR2GRAY, this->imgCanny, 200, 255);
        

        morphologyEx(this->imgCanny, this->imgCanny, MORPH_CLOSE, this->morph_kernel, Point(-1,-1), 1);

                
        this->hullBoat = findContour(this->imgCanny);

        float closest = 1000000;

        Rect rectAux;
        Point pointAux;
        
        bool draw = false;
        for ( size_t i = 0; i < hullBoat.size(); i++ ){
            rectAux = boundingRect(hullBoat[i]);
            pointAux.x = (rectAux.x + rectAux.width) - rectAux.width/2;
            pointAux.y = (rectAux.y + rectAux.height) - rectAux.height/2;

            
            float distance = hypot(pointAux.x - this->imgMain.cols/2, pointAux.y - this->imgMain.rows/2);
            float areaBoat = contourArea(hullBoat[i]);

            if(areaBoat >= 1000){
                    closest = distance;
                    this->rectBoat = rectAux;
                    this->pointBoat = pointAux;
                    
                    draw = true;
            }
            
        }
        if(draw){
            rectangle(this->imgMain, this->rectBoat, viz::Color::green(), 3);
            circle(this->imgMain, this->pointBoat, 1, viz::Color::green(), 2);

            this->dxPixel = (float(this->pointBoat.x) - float(this->imgMain.cols/2));
            this->dyPixel = (float(this->imgMain.rows/2) - float(this->pointBoat.y));
            draw = false;
            
        }else{
            
            this->dxPixel = 1000000;
            this->dyPixel = 1000000;
        }
    
        
        
        
        
    }catch(Exception& e){
        this->imgCanny = Mat::ones(this->imgBGR2GRAY_t_1.rows, this->imgBGR2GRAY_t_1.cols, CV_8U);
        this->dxPixel = 1000000;
        this->dyPixel = 1000000;
    }
    

}

void Vision::processImage(){
    circle(this->imgMain, Point(this->imgMain.cols/2, this->imgMain.rows/2), 1, viz::Color::black(), 3);
    findLight();
    if(!this->firstView){
        findBoat();
    }
    
    this->firstView = false;
    imshow("camera", this->imgMain);
    waitKey(1);
    
    
}






































