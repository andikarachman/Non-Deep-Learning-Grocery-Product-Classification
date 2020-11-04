/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <filesystem>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "matching2D.hpp"

using namespace std;
using namespace cv;
namespace fs = std::__fs::filesystem;


int main(int argc, char** argv)
{
    /**************************************/
    /* INIT VARIABLES AND DATA STRUCTURES */
    /**************************************/
        
    // data location
    int imgStartIndex = 0;   // first file index to load 
    int imgEndIndex = 10;   // last file index to load

    // detector and descriptor config
    string detectorType = "SIFT";            // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT  
    string descriptorType = "SIFT";          // BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK
    string kptPath = "../ref/keypoints/";
    string dscPath = "../ref/descriptors/";

    // matcher config
    string matcherType = "MAT_FLANN";         // MAT_BF, MAT_FLANN
    string matcherDescriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";          // SEL_NN, SEL_KNN

    // multithreading config 
    const size_t nthreads = thread::hardware_concurrency();
    vector<thread> threads(nthreads);
    vector<string> kptFiles;
    vector<string> dscFiles;
    for (const auto& kpt : fs::directory_iterator(kptPath)) 
    {
        string kptFile = kpt.path().string();
        if (kptFile.substr(kptFile.length()-3, 3) != "txt") 
        {
            continue;
        }
        kptFiles.push_back(kptFile);
            
        // extract product name
        std::string delimiter = "/";
        std::string token;
        std::string productName = kptFile.substr(0, kptFile.find(".txt"));
        size_t pos = 0;
        while ((pos = productName.find(delimiter)) != std::string::npos) 
        {
            token = productName.substr(0, pos);
            productName.erase(0, pos + delimiter.length());
        }
        string dscFile = dscPath + productName + ".xml";
        dscFiles.push_back(dscFile);
    }

    cv::VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open(0)) 
    {
        return 0;
    }
        
    for(;;)
    {
        cv::Mat frame;
        cap >> frame;
        cv::resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC);

        // result
        double maxMatches = -1.0;
        string closestImg = "None";
          
        double t = (double)cv::getTickCount();

        /***************************/
        /* PROCESSING SOURCE IMAGE */
        /***************************/

        // load source image and convert it to grayscale
        cv::Mat srcImg, srcImgGray;
        cv::cvtColor(frame, srcImgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from the source image
        vector<cv::KeyPoint> srcKeypoints; // create empty feature list for the source image
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(srcKeypoints, srcImgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(srcKeypoints, srcImgGray, false);
        }
        else
        {
            detKeypointsModern(srcKeypoints, srcImgGray, detectorType, false);
        }

        // extract keypoint descriptors of the source image
        cv::Mat srcDescriptors;
        descKeypoints(srcKeypoints, srcImgGray, srcDescriptors, descriptorType);

        
        /***************************************/
        /* MAIN LOOP OVER ALL REFERENCE IMAGES */
        /***************************************/

        for(int t = 0; t < nthreads; t++) 
        {   // threading loop
            threads[t] = std::thread(std::bind(
            [&](const int bi, const int ei, const int t)
            {

            for (size_t imgIndex = bi; imgIndex <= ei; imgIndex++)
            {   // image loop

                /**************************/
                /* LOAD IMAGE INTO BUFFER */
                /**************************/

                // assemble filenames for current index
                // ostringstream imgNumber;
                // imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                // string refImgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load reference image from file and convert to grayscale
                //cv::Mat refImg, refImgGray;
                // refImg = cv::imread(refImgFullFilename);
                // cv::cvtColor(refImg, refImgGray, cv::COLOR_BGR2GRAY);


                /**************************/
                /* DETECT IMAGE KEYPOINTS */
                /**************************/
        
                // import keypoints from ref file
                vector<cv::KeyPoint> refKeypoints;
                ifstream infile;
                string temp;
                string txtName = kptFiles.at(imgIndex);
                // string txtName = productName + ".txt";
                infile.open(txtName);
                while (getline(infile, temp))
                {
                    stringstream ss(temp);
                    istream_iterator<string> begin(ss);
                    istream_iterator<string> end;
                    vector<string> vstrings(begin, end);
                    
                    cv::KeyPoint kpt;
                    kpt.pt = cv::Point2f(stof(vstrings[0]), stof(vstrings[1]));
                    kpt.size = stof(vstrings[2]);
                    kpt.response = stof(vstrings[3]);
                    kpt.octave = stoi(vstrings[4]);
                    kpt.angle = stof(vstrings[5]);

                    refKeypoints.push_back(kpt);
                }
                infile.close();

                // extract product name
                std::string delimiter = "/";
                std::string token;
                std::string productName = txtName.substr(0, txtName.find(".txt"));
                size_t pos = 0;
                while ((pos = productName.find(delimiter)) != std::string::npos) {
                    token = productName.substr(0, pos);
                    productName.erase(0, pos + delimiter.length());
                }

                /********************************/
                /* EXTRACT KEYPOINT DESCRIPTORS */
                /********************************/

                // import descriptors from ref file
                cv::Mat refDescriptors;
                string xmlName = dscFiles.at(imgIndex);
                cv::FileStorage file(xmlName, cv::FileStorage::READ);
                file["descriptor"] >> refDescriptors;
                

                /*****************************************************************/
                /* MATCH KEYPOINT DESCRIPTORS BETWEEN SOURCE AND REFERENCE IMAGE */
                /*****************************************************************/

                vector<cv::DMatch> matches;
                matchDescriptors(srcKeypoints, refKeypoints, srcDescriptors, refDescriptors,
                                matches, matcherDescriptorType, matcherType, selectorType);

                if (((double)matches.size()/refKeypoints.size()) > maxMatches)
                {
                    maxMatches = (double)matches.size() / refKeypoints.size();
                    if (maxMatches < 0.05 && (productName != "chitato" || productName != "lays")) 
                    {
                        closestImg = "None";
                    }
                    else 
                    {
                        closestImg = productName;
                    }
                    
                }

            } // eof loop over all images
            
            }
            ,t * (imgEndIndex - imgStartIndex) / nthreads
            ,(t+1) == nthreads ? (imgEndIndex - imgStartIndex) : 
            (t+1) * (imgEndIndex - imgStartIndex) / nthreads, t));
        }

        
        for_each(threads.begin(),threads.end(),[](std::thread& x)
        {
            x.join();
        });
        
        // Results
        cout << "Product: " << closestImg << endl;
        cout << "Score: " << maxMatches << endl;
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "Product classification elapsed time in " << t << " s" << endl;
        
        if (frame.empty()) 
        {
             break; // end of video stream
        }
        
        cv::putText(frame, closestImg, 
            cv::Point(10, frame.rows / 2), //top-left position
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            CV_RGB(252, 94, 3), //font color
            2);

        bool visInfo = true;
        if (visInfo && closestImg == "pocky_choco")
        {
            cv::putText(frame, "Kategory: Snack", 
            cv::Point(10, frame.rows / 2 + 25), //top-left position
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            CV_RGB(3, 102, 252), //font color
            2);

            cv::putText(frame, "Harga: IDR 10.000", 
            cv::Point(10, frame.rows / 2 + 50), //top-left position
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            CV_RGB(3, 102, 252), //font color
            2);

            cv::putText(frame, "Energi Total: 110 kkal", 
            cv::Point(10, frame.rows / 2 + 75), //top-left position
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            CV_RGB(3, 102, 252), //font color
            2);
        }

        cv::imshow("GetGO Product Classification", frame);
        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 

        // Reset results
        maxMatches = -1.0;
        closestImg = "None";
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}

