#include "Header.h"

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ctime>
#include <iostream>
#include <time.h>
#include <windows.h>


using namespace std;
using namespace NTUST_MSPLab;

FBRLF::FBRLF(void)
{
	testLocations = vector< pair<cv::Point2i, cv::Point2i> >(DESC_LEN);
	integralImage = cvCreateMat(1, 1, CV_32SC1);
	pickTestLocations();
}

FBRLF::~FBRLF(void)
{
	testLocations.clear();
	cvReleaseMat(&integralImage);
}

void FBRLF::getFBRLFDescriptor(bitset<DESC_LEN>& desc, cv::KeyPoint kpt, IplImage* img)
{
	int inWS = integralImage->step / CV_ELEM_SIZE(integralImage->type);

	int* iD = integralImage->data.i;

	for (int i = 0; i < DESC_LEN; ++i) {

		const pair<cv::Point2i, cv::Point2i>& tL = testLocations[i];

		const cv::Point2i p1 = CV_POINT_PLUS(kpt.pt, tL.first);
		const cv::Point2i p2 = CV_POINT_PLUS(kpt.pt, tL.second);

		const int intensity1 =
			GET_PIXEL_NW(iD, p1, inWS) - GET_PIXEL_NE(iD, p1, inWS) -
			GET_PIXEL_SW(iD, p1, inWS) + GET_PIXEL_SE(iD, p1, inWS);

		const int intensity2 =
			GET_PIXEL_NW(iD, p2, inWS) - GET_PIXEL_NE(iD, p2, inWS) -
			GET_PIXEL_SW(iD, p2, inWS) + GET_PIXEL_SE(iD, p2, inWS);
		desc[i] = intensity1 < intensity2;
	}
}

void FBRLF::getFBRLFDescriptors(vector< bitset<DESC_LEN> >& descriptors, const vector<cv::KeyPoint>& kpts,
	IplImage* img)
{
	assert(img->nChannels == 1);

	assert(validateKeypoints(kpts, img->width, img->height));

	descriptors.resize(kpts.size());

	allocateIntegralImage(img);

	cvIntegral(img, integralImage);
  
	for (unsigned int i = 0; i < kpts.size(); ++i)
	{	
		FILETIME ftBeg, ftEnd;
		GetSystemTimeAsFileTime(&ftBeg);
		for (int a=0;a<1000;a++)
			getFBRLFDescriptor(descriptors[i], kpts[i], img);
		GetSystemTimeAsFileTime(&ftEnd);
		printf("%lf secs \n", (ftEnd.dwLowDateTime - ftBeg.dwLowDateTime) * 1E-7);
		Sleep(12);
	}
}

void FBRLF::pickTestLocations(void)
{
	for (int i = 0; i < DESC_LEN; ++i) {
		pair<cv::Point2i, cv::Point2i>& tL = testLocations[i];

		tL.first.x = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
		tL.first.y = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
		tL.second.x = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
		tL.second.y = int(rand() % PATCH_SIZE) - HALF_PATCH_SIZE - 1;
	}
}

bool FBRLF::isKeypointInsideSubImage(const cv::KeyPoint& kpt, const int width, const int height)
{
	return
		SUBIMAGE_LEFT <= kpt.pt.x  &&  kpt.pt.x < SUBIMAGE_RIGHT(width) &&
		SUBIMAGE_TOP <= kpt.pt.y  &&  kpt.pt.y < SUBIMAGE_BOTTOM(height);
}

bool FBRLF::validateKeypoints(const vector<cv::KeyPoint>& kpts, int im_w, int im_h)
{
	for (unsigned int i = 0; i < kpts.size(); ++i)
		if (!isKeypointInsideSubImage(kpts[i], im_w, im_h))
			return false;

	return true;
}

void FBRLF::allocateIntegralImage(const IplImage* img)
{
	const int im_w_1 = img->width + 1, im_h_1 = img->height + 1;

	if (im_w_1 != integralImage->width && im_h_1 != integralImage->height) {
		cvReleaseMat(&integralImage);
		integralImage = cvCreateMat(im_h_1, im_w_1, CV_32SC1);
	}
}
