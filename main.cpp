#include "function.h"
#include "Header.h"
#include <cstdio>
#include "opencv2/calib3d/calib3d.hpp"
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#pragma warning( disable : 4996 2664 )  
using namespace std;
using namespace cv;


static const int Image_WIDTH = 800;
static const int Image_HEIGHT = 640;

static const int MAXIMUM_NUMBER_OF_MATCHES = 3000;

static const float SMALLEST_SCALE_CHANGE = 0.5;

static const int NUMBER_OF_SCALE_STEPS = 3;

static const int NUMBER_OF_ROTATION_STEPS = 20; //Key---­«­n

NTUST_MSPLab::FBRLF FBRLF;

vector<cv::KeyPoint> templateKpts[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

vector< bitset<NTUST_MSPLab::DESC_LEN> > templateDescs[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

CvMat templateObjectCorners[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

double* templateObjectCornersData[NUMBER_OF_SCALE_STEPS][NUMBER_OF_ROTATION_STEPS];

double match1Data[2 * MAXIMUM_NUMBER_OF_MATCHES]; //represents the  keypoint coordinates  of the matching  template
double match2Data[2 * MAXIMUM_NUMBER_OF_MATCHES]; // represents  the  matching  keypoints  detected  in  the current frame

bool doDBSearch = true;

IplImage* templateImageGrayFull;

IplImage* templateImageGray = NULL;

IplImage* newFrameGray;

IplImage* outputImage;

IplImage* sideBySideImage;

int fastThreshold;

int templateROIX;
int templateROIY;

double pointsOnOriginalImage[MAXIMUM_NUMBER_OF_MATCHES];

int main(int argc, char** argv) {

	Initial();

	Start(argc, argv);

	return 0;
}


void Initial() {
	srand(time(NULL));

	newFrameGray = cvCreateImage(cvSize(Image_WIDTH, Image_HEIGHT), IPL_DEPTH_8U, 1);
	outputImage = cvCreateImage(cvSize(Image_WIDTH, Image_HEIGHT), IPL_DEPTH_8U, 1);
	templateImageGrayFull = cvCreateImage(cvSize(Image_WIDTH, Image_HEIGHT), IPL_DEPTH_8U, 1);
	sideBySideImage = cvCreateImage(cvSize(2 * Image_WIDTH, Image_HEIGHT), IPL_DEPTH_8U, 1);
	templateImageGray = cvCreateImage(cvSize(1, 1), IPL_DEPTH_8U, 1);
	for (int s = 0; s < NUMBER_OF_SCALE_STEPS; s++) {
		for (int r = 0; r < NUMBER_OF_ROTATION_STEPS; r++) {
			templateObjectCornersData[s][r] = new double[8];
			templateObjectCorners[s][r] = cvMat(1, 4, CV_64FC2, templateObjectCornersData[s][r]);
		}
	}
}

void Start(int argc, char** argv) {
	float TP = 0;
	float FP = 0;
	float TN = 0;
	float FN = 0;
	IplImage* result = outputImage;
	int matchnumber, number, matchcompare_number, testcompare_number, totalmatchnumber = 0;
	char  str1[30], str2[20], str3[20];
	ofstream matchfile, matchfile_num;

	time_t t1, t2;
	//char *folders[5] = {"bark","bikes","boat","graf","leuven"};
	/*char *folders[5] = { "boat","graf","leuven" };*/
	char *folders[5] = {"graf"};
	for (int outter = 0; outter < 3; outter++)
	{
		for (int inner = 1; inner < 7; inner++)
		{
			cout << "now matching..  ";
			char str_root[] = "./dataset/";
			strcat(str_root,folders[outter]);
			strcat(str_root,"/img");
			char temp[2];
			itoa(inner, temp, 10);
			strcat(str_root, temp);
			strcat(str_root, ".tif");

			newFrameGray = cvLoadImage(str_root, 0);
			//clock_t t1, t2;
			//t1 = clock();
			/*cv::Mat newFrameGray_tmp = cv::cvarrToMat(newFrameGray);
			resize(newFrameGray_tmp, newFrameGray_tmp, Size(Image_WIDTH, Image_HEIGHT));
			newFrameGray = cvCloneImage(&(IplImage)newFrameGray_tmp);*/
			cout << str_root << " and ";
			number = atoi(argv[2]);
			FeatureDetect();

			for (int numbercount = 1; numbercount < number; numbercount++) {
				if (numbercount == inner)
					continue;
				char str_cop[] = "./dataset/";
				strcat(str_cop, folders[outter]);
				strcat(str_cop, "/img");
				_itoa(numbercount, str2, 10);
				strcat(str_cop, str2);
				strcat(str_cop, ".tif");
				newFrameGray = cvLoadImage(str_cop, 0);
				//cv::Mat newFrameGray_tmp = cv::cvarrToMat(newFrameGray);
				//resize(newFrameGray_tmp, newFrameGray_tmp, Size(Image_WIDTH, Image_HEIGHT));
				//newFrameGray = cvCloneImage(&(IplImage)newFrameGray_tmp);
				cout << str_cop << endl;
				/*PutImagesSideBySide(sideBySideImage, newFrameGray, templateImageGrayFull);*/
				matchnumber = DoDetection();
				//t2 = clock();
				//printf("%lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));
				totalmatchnumber = totalmatchnumber + matchnumber;
				cvReleaseImage(&newFrameGray);
				result = sideBySideImage;
				
				if (matchnumber > 4)
					TP += 1;
				else
					FN += 1;
			}
      
			for (int i = 0; i < 3; i++)
			{
				if (i == (outter))
					continue;
				for (int numbercount = 1; numbercount < number; numbercount++) {
					char str_cop[] = "./dataset/";
					/*strcat(str_cop, folders[i]);*/
					strcat(str_cop, folders[i]);
					strcat(str_cop, "/img");
					_itoa(numbercount, str3, 10);
					strcat(str_cop, str3);
					strcat(str_cop, ".tif");
					newFrameGray = cvLoadImage(str_cop, 0);
					cout << str_cop << endl;
					/*cv::Mat newFrameGray_tmp = cv::cvarrToMat(newFrameGray);
					resize(newFrameGray_tmp, newFrameGray_tmp, Size(Image_WIDTH, Image_HEIGHT));
					newFrameGray = cvCloneImage(&(IplImage)newFrameGray_tmp);*/
					PutImagesSideBySide(sideBySideImage, newFrameGray, templateImageGrayFull);
					matchnumber = DoDetection();
					totalmatchnumber = totalmatchnumber + matchnumber;
					cvReleaseImage(&newFrameGray);
					result = sideBySideImage;
					if (matchnumber > 4)
						FP += 1;
					else
						TN += 1;
				}
			}
		}
	}
	
	cout << "TP :" << TP << endl;
	cout << "FN :" << FN << endl;
	cout << "FP :" << FP << endl;
	cout << "TN :" << TN << endl;
	cout << "ACC :" << (1-(TP / (TP + FP))) << endl;
	cout << "Rec :" << (TP / (TP + FN)) << endl;

	waitKey(0);

	//if (totalmatchnumber > 5) {
	//	matchfile.open("matchfile2.txt");
	//	matchfile << "1\n";
	//	matchfile.close();
	//	matchfile_num.open("matchfilenumber2.txt");
	//	matchfile_num << totalmatchnumber;
	//	matchfile_num.close();
	//}
	//else {
	//	matchfile.open("matchfile2.txt");
	//	matchfile << "0\n";
	//	matchfile.close();
	//	matchfile_num.open("matchfilenumber2.txt");
	//	matchfile_num << totalmatchnumber;
	//	matchfile_num.close();
	//}
}

void FeatureDetect() {
	cvCopy(newFrameGray, outputImage);
	SaveNewTemplate();
	FBRLFFeature();
}

bool SaveNewTemplate() {
	const CvSize templateSize = cvSize(Image_WIDTH, Image_HEIGHT);
	cvCopy(newFrameGray, templateImageGrayFull);
	cvReleaseImage(&templateImageGray);
	templateImageGray = cvCreateImage(templateSize, IPL_DEPTH_8U, 1);
	cvCopy(newFrameGray, templateImageGray);
	SaveCornerCoors();
	return true;

}

void SaveCornerCoors() {
	const double templateWidth = templateImageGray->width;
	const double templateHeight = templateImageGray->height;

	double* corners = templateObjectCornersData[0][0];
	corners[0] = 0;
	corners[1] = 0;
	corners[2] = templateWidth;
	corners[3] = 0;
	corners[4] = templateWidth;
	corners[5] = templateHeight;
	corners[6] = 0;
	corners[7] = templateHeight;
}
void FBRLFFeature() {
	static const float ROT_ANGLE_INCREMENT = 360.0 / NUMBER_OF_ROTATION_STEPS;
	static const float k = exp(log(SMALLEST_SCALE_CHANGE) / (NUMBER_OF_SCALE_STEPS - 1));

	fastThreshold = ChooseFASTThreshold(templateImageGray, 200,300);

	for (int scaleInd = 0; scaleInd < NUMBER_OF_SCALE_STEPS; ++scaleInd) {

		const float currentScale = pow(k, scaleInd);
		IplImage* scaledTemplateImg = cvCreateImage(cvSize(templateImageGray->width * currentScale, templateImageGray->height * currentScale), IPL_DEPTH_8U, 1);
		cvResize(templateImageGray, scaledTemplateImg);

		const CvPoint2D32f center = cvPoint2D32f(scaledTemplateImg->width >> 1, scaledTemplateImg->height >> 1);

		float currentAngle = 0.0;
		for (int rotInd = 0; rotInd < NUMBER_OF_ROTATION_STEPS; ++rotInd, currentAngle += ROT_ANGLE_INCREMENT) {
			IplImage* rotatedImage = cvCreateImage(cvGetSize(scaledTemplateImg), scaledTemplateImg->depth, scaledTemplateImg->nChannels);
			RotateImage(rotatedImage, scaledTemplateImg, center, -currentAngle);
			ExtractKeypoints(templateKpts[scaleInd][rotInd], fastThreshold, rotatedImage);

			FBRLF.getFBRLFDescriptors(templateDescs[scaleInd][rotInd], templateKpts[scaleInd][rotInd], rotatedImage);

			EstimateCornerCoordinatesOfNewTemplate(scaleInd, rotInd, currentScale, currentAngle);

			cvReleaseImage(&rotatedImage);
		}
		cvReleaseImage(&scaledTemplateImg);
	}
}

int ChooseFASTThreshold(const IplImage* img, const int lowerBound, const int upperBound) {
	static vector<cv::KeyPoint> kpts;

	int left = 0;
	int right = 255;
	int currentThreshold = 128;
	int currentScore = 256;

	IplImage* copyImg = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	cvCopy(img, copyImg);
	currentScore = ExtractKeypoints(kpts, currentThreshold, copyImg);
	while (currentScore < lowerBound || currentScore > upperBound) {
		currentScore = ExtractKeypoints(kpts, currentThreshold, copyImg);

		if (lowerBound > currentScore) {

			right = currentThreshold;
			currentThreshold = (currentThreshold + left) >> 1;
			if (right == currentThreshold)
				break;
		}
		else {

			left = currentThreshold;
			currentThreshold = (currentThreshold + right) >> 1;
			if (left == currentThreshold)
				break;
		}
	}
	cvReleaseImage(&copyImg);

	return currentThreshold;
}

int ExtractKeypoints(vector< cv::KeyPoint >& kpts, int kptDetectorThreshold, IplImage* img) {
	CvRect r = cvRect(NTUST_MSPLab::IMAGE_PADDING_LEFT, NTUST_MSPLab::IMAGE_PADDING_TOP,
		NTUST_MSPLab::SUBIMAGE_WIDTH(img->width), NTUST_MSPLab::SUBIMAGE_HEIGHT(img->height));

	cvSetImageROI(img, r);

	//cv::Mat imageinROI(img, 0);
	cv::Mat imageinROI = cv::cvarrToMat(img);

	cv::AGAST(imageinROI, kpts, kptDetectorThreshold, true);
  
	cvResetImageROI(img);

	for (unsigned int i = 0, sz = kpts.size(); i < sz; ++i)
		kpts[i].pt.x += NTUST_MSPLab::IMAGE_PADDING_LEFT, kpts[i].pt.y += NTUST_MSPLab::IMAGE_PADDING_TOP;

	return kpts.size();
}

void RotateImage(IplImage* dst, IplImage* src, const CvPoint2D32f& center, float angle) {
	static CvMat *rotMat = cvCreateMat(2, 3, CV_32FC1);
	cv2DRotationMatrix(center, angle, 1.0, rotMat);
	cvWarpAffine(src, dst, rotMat);
}

void EstimateCornerCoordinatesOfNewTemplate(int scaleInd, int rotInd, float scale, float angle) {
	static double* corners = templateObjectCornersData[0][0];

	const float orgCenterX = templateImageGray->width / 2.0, orgCenterY = templateImageGray->height / 2.0;
  
	const float centerX = orgCenterX * scale, centerY = orgCenterY * scale;

	const float cosAngle = cos(DegreeToRadian(angle));
	const float sinAngle = sin(DegreeToRadian(angle));

	for (int xCoor = 0, yCoor = 1; xCoor < 8; xCoor += 2, yCoor += 2) {
		const float resizedAndTranslatedX = (corners[xCoor] * scale) - centerX,
			resizedAndTranslatedY = (corners[yCoor] * scale) - centerY;

		templateObjectCornersData[scaleInd][rotInd][xCoor] = (resizedAndTranslatedX * cosAngle - resizedAndTranslatedY * sinAngle) + centerX;
		templateObjectCornersData[scaleInd][rotInd][yCoor] = (resizedAndTranslatedX * sinAngle + resizedAndTranslatedY * cosAngle) + centerY;
	}
}
inline float DegreeToRadian(const float d) {
	return (d / 180.0) * 3.14;
}

void PutImagesSideBySide(IplImage* result, const IplImage* img1, const IplImage* img2) {

	const int bigWS = result->widthStep;

	const int bigHalfWS = result->widthStep >> 1;

	const int lWS = img1->widthStep;

	const int rWS = img2->widthStep;

	char *p_big = result->imageData;

	char *p_bigMiddle = result->imageData + bigHalfWS;

	const char *p_l = img1->imageData;

	const char *p_r = img2->imageData;

	for (int i = 0; i < Image_HEIGHT; ++i, p_big += bigWS, p_bigMiddle += bigWS) {

		memcpy(p_big, p_l + i * lWS, lWS);
		memcpy(p_bigMiddle, p_r + i * rWS, rWS);
	}
}

int DoDetection() {

	static int64 startTime, endTime;

	// Homography Matrix
	static CvMat* H = cvCreateMat(3, 3, CV_64FC1);

	static double detectedObjCornersData[8];
	static CvMat detectedObjCorners = cvMat(1, 4, CV_64FC2, detectedObjCornersData);

	vector<cv::KeyPoint> kpts;

	vector< bitset<NTUST_MSPLab::DESC_LEN> > descs;


	CvMat match1, match2;

	float maxRatio = 0.0;
	int maxScaleInd = 0;
	int maxRotInd = 0;
	int maximumNumberOfMatches = 0;
	int matchmark = 0;

	const int dbScaleSz = doDBSearch ? NUMBER_OF_SCALE_STEPS : 1;
	const int dbRotationSz = doDBSearch ? NUMBER_OF_ROTATION_STEPS : 1;

	ExtractKeypoints(kpts, fastThreshold, newFrameGray);

	FBRLF.getFBRLFDescriptors(descs, kpts, newFrameGray);

	for (int scaleInd = 0; scaleInd < dbScaleSz; ++scaleInd) {
		for (int rotInd = 0; rotInd < dbRotationSz; ++rotInd) {
			const int numberOfMatches = MatchDescriptors(match1, match2, templateDescs[scaleInd][rotInd], descs, templateKpts[scaleInd][rotInd], kpts);

			if (numberOfMatches < 4)
				continue;

			const float currentRatio = float(numberOfMatches) / templateKpts[scaleInd][rotInd].size();
			if (currentRatio > maxRatio) {
				maxRatio = currentRatio;
				maxScaleInd = scaleInd;
				maxRotInd = rotInd;
				maximumNumberOfMatches = numberOfMatches;
			}
		}
	}


	if (maximumNumberOfMatches > 3) {
		MatchDescriptors(match1, match2, templateDescs[maxScaleInd][maxRotInd], descs, templateKpts[maxScaleInd][maxRotInd], kpts);

		cvFindHomography(&match1, &match2, H, CV_RANSAC, 10, 0);

		if (NiceHomography(H)) {

			cvPerspectiveTransform(&templateObjectCorners[maxScaleInd][maxRotInd], &detectedObjCorners, H);

			//matchmark=MarkDetectedObject(sideBySideImage, detectedObjCornersData);

			TransformPointsIntoOriginalImageCoordinates(maximumNumberOfMatches, maxScaleInd, maxRotInd);
		}
	}
	//ShowKeypoints(sideBySideImage, kpts);
	//ShowMatches(maximumNumberOfMatches);
	//if (matchmark == 0)
	//{
	//	maximumNumberOfMatches = 0;
	//}

	return maximumNumberOfMatches;
}

int MatchDescriptors(CvMat& match1, CvMat& match2, const vector< bitset<NTUST_MSPLab::DESC_LEN> > descs1,
	const vector< bitset<NTUST_MSPLab::DESC_LEN> > descs2, const vector<cv::KeyPoint>& kpts1, const vector<cv::KeyPoint>& kpts2) {

	static const int MAX_MATCH_DISTANCE = 21;

	int numberOfMatches = 0;

	int bestMatchInd2 = 0;

	for (unsigned int i = 0; i < descs1.size() && numberOfMatches < MAXIMUM_NUMBER_OF_MATCHES; ++i) {
		int minDist = NTUST_MSPLab::DESC_LEN;
		for (unsigned int j = 0; j < descs2.size(); ++j) {
			const int dist = NTUST_MSPLab::HAMMING_DISTANCE(descs1[i], descs2[j]);
			if (dist < minDist) {
				minDist = dist;
				bestMatchInd2 = j;
			}

		}
		if (minDist > MAX_MATCH_DISTANCE)
			continue;
		const int xInd = 2 * numberOfMatches;
		const int yInd = xInd + 1;

		match1Data[xInd] = kpts1[i].pt.x;
		match1Data[yInd] = kpts1[i].pt.y;

		match2Data[xInd] = kpts2[bestMatchInd2].pt.x;
		match2Data[yInd] = kpts2[bestMatchInd2].pt.y;

		numberOfMatches++;
	}

	if (numberOfMatches > 0) {
		cvInitMatHeader(&match1, numberOfMatches, 2, CV_64FC1, match1Data);
		cvInitMatHeader(&match2, numberOfMatches, 2, CV_64FC1, match2Data);
	}

	return numberOfMatches;
}

bool NiceHomography(const CvMat * H) {
	const double det = cvmGet(H, 0, 0) * cvmGet(H, 1, 1) - cvmGet(H, 1, 0) * cvmGet(H, 0, 1);
	if (det < 0)
		return false;

	const double N1 = sqrt(cvmGet(H, 0, 0) * cvmGet(H, 0, 0) + cvmGet(H, 1, 0) * cvmGet(H, 1, 0));
	if (N1 > 4 || N1 < 0.1)
		return false;

	const double N2 = sqrt(cvmGet(H, 0, 1) * cvmGet(H, 0, 1) + cvmGet(H, 1, 1) * cvmGet(H, 1, 1));
	if (N2 > 4 || N2 < 0.1)
		return false;

	const double N3 = sqrt(cvmGet(H, 2, 0) * cvmGet(H, 2, 0) + cvmGet(H, 2, 1) * cvmGet(H, 2, 1));
	if (N3 > 0.002)
		return false;

	return true;
}

int MarkDetectedObject(IplImage* frame, const double * detectedCorners) {
	DrawQuadrangle(frame,
		detectedCorners[0], detectedCorners[1],
		detectedCorners[2], detectedCorners[3],
		detectedCorners[4], detectedCorners[5],
		detectedCorners[6], detectedCorners[7],
		cvScalar(255, 255, 255), 3);
	return 1;
}

void DrawQuadrangle(IplImage* frame,
	const int u0, const int v0,
	const int u1, const int v1,
	const int u2, const int v2,
	const int u3, const int v3,
	const CvScalar color, const int thickness)
{
	cvLine(frame, cvPoint(u0, v0), cvPoint(u1, v1), color, thickness);
	cvLine(frame, cvPoint(u1, v1), cvPoint(u2, v2), color, thickness);
	cvLine(frame, cvPoint(u2, v2), cvPoint(u3, v3), color, thickness);
	cvLine(frame, cvPoint(u3, v3), cvPoint(u0, v0), color, thickness);
}

void TransformPointsIntoOriginalImageCoordinates(const int matchNo, const int scaleInd, const int rotInd) {

	static const float ROT_ANGLE_INCREMENT = 360.0 / NUMBER_OF_ROTATION_STEPS;

	static const float k = exp(log(SMALLEST_SCALE_CHANGE) / (NUMBER_OF_SCALE_STEPS - 1));
	const float scale = pow(k, scaleInd);
	const float orgCenterX = templateImageGray->width / 2.0;
	const float orgCenterY = templateImageGray->height / 2.0;

	const float centerX = orgCenterX * scale;
	const float centerY = orgCenterY * scale;

	const float angle = ROT_ANGLE_INCREMENT * rotInd;

	const float cosAngle = cos(DegreeToRadian(-angle));
	const float sinAngle = sin(DegreeToRadian(-angle));
	const float iterationEnd = 2 * matchNo;
	for (int xCoor = 0, yCoor = 1; xCoor < iterationEnd; xCoor += 2, yCoor += 2) {

		const float translatedX = match1Data[xCoor] - centerX;
		const float translatedY = match1Data[yCoor] - centerY;

		const float rotatedBackX = translatedX * cosAngle - translatedY * sinAngle;
		const float rotatedBackY = translatedX * sinAngle + translatedY * cosAngle;

		pointsOnOriginalImage[xCoor] = rotatedBackX / scale + orgCenterX;
		pointsOnOriginalImage[yCoor] = rotatedBackY / scale + orgCenterY;

	}
}

void ShowKeypoints(IplImage* img, const vector<cv::KeyPoint>& kpts) {
	for (unsigned int i = 0; i < kpts.size(); ++i)
		DrawAPlus(img, kpts[i].pt.x, kpts[i].pt.y);
}

void DrawAPlus(IplImage* img, const int x, const int y) {
	cvLine(img, cvPoint(x - 5, y), cvPoint(x + 5, y), CV_RGB(255, 255, 255));
	cvLine(img, cvPoint(x, y - 5), cvPoint(x, y + 5), CV_RGB(255, 255, 255));
}
void ShowMatches(const int matchCount) {
	const int iterationEnd = 2 * matchCount;

	for (int xCoor = 0, yCoor = 1; xCoor < iterationEnd; xCoor += 2, yCoor += 2) {

		cvLine(sideBySideImage,
			cvPoint(match2Data[xCoor], match2Data[yCoor]),
			cvPoint(pointsOnOriginalImage[xCoor] + templateROIX + Image_WIDTH,
				pointsOnOriginalImage[yCoor] + templateROIY),
			cvScalar(255, 255, 255), 1);
	}
}
