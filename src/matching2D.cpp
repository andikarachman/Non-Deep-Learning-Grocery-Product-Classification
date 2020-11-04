#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType)
{
    double t = (double)cv::getTickCount();
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to 
          // floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to 
          // floating point due to a bug in current OpenCV implementation
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    } 
    else 
    {
        cout << "ERROR in matcher-type within matchDescriptors() ....exitting" << std::endl;
        exit(EXIT_FAILURE);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2;
        double distRatio = 0.8;

        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        // distance ratio filtering
        for (int i = 0; i < kPtsSource.size(); ++i) 
        {
            if (knn_matches[i][0].distance < distRatio * knn_matches[i][1].distance) 
            {
              matches.push_back(knn_matches[i][0]);
            }
        }
    }
    else 
    {
        cout << "ERROR in selector-type within matchDescriptors() ....exitting" << std::endl;
        exit(EXIT_FAILURE);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << " Match descriptor in " << 1000 * t / 1.0 << " ms" << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;             // FAST/AGAST detection threshold score.
        int octaves = 3;                // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f;      // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        int nfeatures = 0;              // the number of best features to retain
        int nOctaveLayers = 3;          // the number of layers in each octave
        double contrastThresh = 0.04;   // the contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        double edgeThresh = 10.;        // the threshold used to filter out edge-like features
        double sigma = 1.6;             // the sigma of the Gaussian applied to the input image at the octave #0
        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThresh, edgeThresh, sigma);
    }
    else if (descriptorType.compare("ORB") == 0) 
    {
        int   nfeatures = 500;          // The maximum number of features to retain.
        float scaleFactor = 1.2f;       // Pyramid decimation ratio, greater than 1.
        int   nlevels = 8;              // The number of pyramid levels.
        int   edgeThreshold = 31;       // This is size of the border where the features are not detected.
        int   firstLevel = 0;           // The level of pyramid to put source image to.
        int   WTA_K = 2;                // The number of points that produce each element of the oriented BRIEF descriptor.
        auto  scoreType = cv::ORB::HARRIS_SCORE;    // HARRIS_SCORE / FAST_SCORE algorithm is used to rank features.
        int   patchSize = 31;           // Size of the patch used by the oriented BRIEF descriptor.
        int   fastThreshold = 20;       // The FAST threshold.
        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                    firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        // Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT,
        //                                   DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int   descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
        int   descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3).
        float threshold = 0.001f;         //   Detector response threshold to accept point.
        int   nOctaves = 4;               // Maximum octave evolution of the image.
        int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
        auto  diffusivity = cv::KAZE::DIFF_PM_G2;   // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
                                                    // DIFF_WEICKERT or DIFF_CHARBONNIER.
        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                      threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (descriptorType.compare("BRIEF") == 0) 
    {
        int bytes = 32;
        bool use_orientation = false;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (descriptorType.compare("FREAK") == 0) 
    {

      bool	orientationNormalized = true;  // Enable orientation normalization.
      bool	scaleNormalized = true;        // Enable scale normalization.
      float patternScale = 22.0f;          // Scaling of the description pattern.
      int	nOctaves = 4;                  // Number of octaves covered by the detected keypoints.
      const std::vector< int > & selectedPairs = std::vector< int >(); // (Optional) user defined selected pairs indexes,

      extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale,
                                               nOctaves, selectedPairs);
    }
    else 
    {
        std::cout << "ERROR in descriptor-type within descKeypoints() ....exitting" << std::endl;
        exit(EXIT_FAILURE);
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using Harris Corner Detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Set detector parameter
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;

    // Apply detector and normalize output
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1); 
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    double t = (double)cv::getTickCount();

    // Normalize output
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (int j = 0; j < dst_norm.rows; j++)
    {
        for (int i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse) 
            { // only store points above a threshold
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                       // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint;  // replace old key point with new one
                            break;              // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


// Detect keypoints in image using moder keypoint detector
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, bool bVis)
{
    // select appropriate detector
    cv::Ptr<cv::FeatureDetector> detector;
    double t = (double)cv::getTickCount();
    if (detectorType == "FAST")
    {
        int threshold = 30;             // difference between intensity of the central pixel and pixels of a circle around
        bool bNMS = true;               // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if (detectorType == "SIFT")
    {
        int nfeatures = 0;              // the number of best features to retain
        int nOctaveLayers = 3;          // the number of layers in each octave
        double contrastThresh = 0.04;   // the contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        double edgeThresh = 10.;        // the threshold used to filter out edge-like features
        double sigma = 1.6;             // the sigma of the Gaussian applied to the input image at the octave #0
        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThresh, edgeThresh, sigma);
    }
    else if (detectorType == "BRISK")
    {
        int threshold = 30;             //   AGAST detection threshold score
        int octaves = 3;                // detection octaves
        float patternScale = 1.0f;      // apply this scale to the pattern used for sampling the neighbourhood of a keypoint
        detector = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (detectorType == "ORB") 
    {
        int   nfeatures = 500;          // The maximum number of features to retain.
        float scaleFactor = 1.2f;       // Pyramid decimation ratio, greater than 1.
        int   nlevels = 8;              // The number of pyramid levels.
        int   edgeThreshold = 31;       // This is size of the border where the features are not detected.
        int   firstLevel = 0;           // The level of pyramid to put source image to.
        int   WTA_K = 2;                // The number of points that produce each element of the oriented BRIEF descriptor.
        auto  scoreType = cv::ORB::HARRIS_SCORE;    // HARRIS_SCORE / FAST_SCORE algorithm is used to rank features.
        int   patchSize = 31;           // Size of the patch used by the oriented BRIEF descriptor.
        int   fastThreshold = 20;       // The FAST threshold.
        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                  firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (detectorType == "AKAZE")
    {
        // Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT,
        //                                   DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        auto  descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int   descriptorSize = 0;        // Size of the descriptor in bits. 0 -> Full size
        int   descriptorChannels = 3;    // Number of channels in the descriptor (1, 2, 3).
        float threshold = 0.001f;         //   Detector response threshold to accept point.
        int   nOctaves = 4;               // Maximum octave evolution of the image.
        int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
        auto  diffusivity = cv::KAZE::DIFF_PM_G2;   // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
                                                    // DIFF_WEICKERT or DIFF_CHARBONNIER.
        detector = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels,
                                     threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else
    {
        cout << "ERROR in detKeypointsModern() ....exitting" << endl;
        exit(EXIT_FAILURE);
    }
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout <<  detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Keypoint Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}