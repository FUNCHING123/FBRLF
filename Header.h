#ifndef __SI_BRIEF_H__
#define __SI_BRIEF_H__

#include <vector>
#include <bitset>
#include <cv.h>
#include <highgui.h>
#pragma warning( disable : 4996 )  
using namespace std;

namespace NTUST_MSPLab {

	static const int DESC_LEN = 256;

	static const int PATCH_SIZE = 37;

	static const int KERNEL_SIZE = 9;

	static const int PATCH_SIZE_2 = PATCH_SIZE * PATCH_SIZE;

	static const int HALF_PATCH_SIZE = PATCH_SIZE >> 1;

	static const int KERNEL_AREA = KERNEL_SIZE * KERNEL_SIZE;

	static const int HALF_KERNEL_SIZE = KERNEL_SIZE >> 1;

	static const int IMAGE_PADDING_TOP = HALF_KERNEL_SIZE + HALF_PATCH_SIZE;
	static const int IMAGE_PADDING_LEFT = IMAGE_PADDING_TOP;
	static const int IMAGE_PADDING_TOTAL = IMAGE_PADDING_TOP << 1;
	static const int IMAGE_PADDING_RIGHT = IMAGE_PADDING_LEFT;
	static const int IMAGE_PADDING_BOTTOM = IMAGE_PADDING_TOP;
	static const int SUBIMAGE_LEFT = IMAGE#ifndef __SI_BRIEF_H__
#define __SI_BRIEF_H__

#include <vector>
#include <bitset>
#include <cv.h>
#include <highgui.h>
#pragma warning( disable : 4996 )  
using namespace std;

namespace NTUST_MSPLab {

	static const int DESC_LEN = 256;

	static const int PATCH_SIZE = 37;
  
	static const int KERNEL_SIZE = 9;
  
	static const int PATCH_SIZE_2 = PATCH_SIZE * PATCH_SIZE;

	static const int HALF_PATCH_SIZE = PATCH_SIZE >> 1;

	static const int KERNEL_AREA = KERNEL_SIZE * KERNEL_SIZE;
  
	static const int HALF_KERNEL_SIZE = KERNEL_SIZE >> 1;

	
	static const int IMAGE_PADDING_TOP = HALF_KERNEL_SIZE + HALF_PATCH_SIZE;
	static const int IMAGE_PADDING_LEFT = IMAGE_PADDING_TOP;
	static const int IMAGE_PADDING_TOTAL = IMAGE_PADDING_TOP << 1;
	static const int IMAGE_PADDING_RIGHT = IMAGE_PADDING_LEFT;
	static const int IMAGE_PADDING_BOTTOM = IMAGE_PADDING_TOP;
	static const int SUBIMAGE_LEFT = IMAGE_PADDING_LEFT;
	static const int SUBIMAGE_TOP = IMAGE_PADDING_TOP;

	inline int HAMMING_DISTANCE(const bitset<DESC_LEN>& d1, const bitset<DESC_LEN>& d2){
		return (d1 ^ d2).count();
	}


	inline int SUBIMAGE_WIDTH(const int width){
		return width - IMAGE_PADDING_TOTAL;
	}

	inline int SUBIMAGE_HEIGHT(const int height){
		return height - IMAGE_PADDING_TOTAL;
	}

	inline int SUBIMAGE_RIGHT(const int width){
		return width - IMAGE_PADDING_RIGHT;
	}
  
	inline int SUBIMAGE_BOTTOM(const int height){
		return height - IMAGE_PADDING_BOTTOM;
	}

	inline int GET_MATRIX_DATA(const int* pD, const int row, int column, const int wS){
		return *(pD + (row * wS) + column);
	}

	inline int GET_PIXEL_NW(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y - HALF_KERNEL_SIZE, point.x - HALF_KERNEL_SIZE, wS);
	}

	inline int GET_PIXEL_NE(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y - HALF_KERNEL_SIZE, point.x + HALF_KERNEL_SIZE, wS);
	}
	inline int GET_PIXEL_SW(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y + HALF_KERNEL_SIZE, point.x - HALF_KERNEL_SIZE, wS);
	}

	inline int GET_PIXEL_SE(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y + HALF_KERNEL_SIZE, point.x + HALF_KERNEL_SIZE, wS);
	}

	inline cv::Point2i CV_POINT_PLUS(const cv::Point2i& p, const cv::Point2i& delta){
		return cv::Point2i(p.x + delta.x, p.y + delta.y);
	}

	class FBRLF {
	public:

		FBRLF(void);

		virtual ~FBRLF();

		void getFBRLFDescriptor(bitset<DESC_LEN>& desc, cv::KeyPoint kpt, IplImage* img);

		void getFBRLFDescriptors(vector< bitset<DESC_LEN> >& descriptors, const vector<cv::KeyPoint>& kpts, IplImage* img);

	private:
		void pickTestLocations(void);

		void allocateIntegralImage(const IplImage* img);

		bool validateKeypoints(const vector< cv::KeyPoint >& kpts, int im_w, int im_h);

		bool isKeypointInsideSubImage(const cv::KeyPoint& kpt, const int width, const int height);

		vector< pair<cv::Point2i, cv::Point2i> > testLocations;

		CvMat* integralImage;
	};

};

#endif_PADDING_LEFT;
	static const int SUBIMAGE_TOP = IMAGE_PADDING_TOP;

	inline int HAMMING_DISTANCE(const bitset<DESC_LEN>& d1, const bitset<DESC_LEN>& d2){
		return (d1 ^ d2).count();
	}

	inline int SUBIMAGE_WIDTH(const int width){
		return width - IMAGE_PADDING_TOTAL;
	}
	inline int SUBIMAGE_HEIGHT(const int height){
		return height - IMAGE_PADDING_TOTAL;
	}
	inline int SUBIMAGE_RIGHT(const int width){
		return width - IMAGE_PADDING_RIGHT;
	}
	inline int SUBIMAGE_BOTTOM(const int height){
		return height - IMAGE_PADDING_BOTTOM;
	}
	inline int GET_MATRIX_DATA(const int* pD, const int row, int column, const int wS){
		return *(pD + (row * wS) + column);
	}

	inline int GET_PIXEL_NW(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y - HALF_KERNEL_SIZE, point.x - HALF_KERNEL_SIZE, wS);
	}

	inline int GET_PIXEL_NE(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y - HALF_KERNEL_SIZE, point.x + HALF_KERNEL_SIZE, wS);
	}

	inline int GET_PIXEL_SW(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y + HALF_KERNEL_SIZE, point.x - HALF_KERNEL_SIZE, wS);
	}

	inline int GET_PIXEL_SE(const int* pD, const cv::Point2i& point, const int wS){
		return GET_MATRIX_DATA(pD, point.y + HALF_KERNEL_SIZE, point.x + HALF_KERNEL_SIZE, wS);
	}
	inline cv::Point2i CV_POINT_PLUS(const cv::Point2i& p, const cv::Point2i& delta){
		return cv::Point2i(p.x + delta.x, p.y + delta.y);
	}
	class FBRLF {
	public:

		FBRLF(void);

		virtual ~FBRLF();
		void getFBRLFDescriptor(bitset<DESC_LEN>& desc, cv::KeyPoint kpt, IplImage* img);
		void getFBRLFDescriptors(vector< bitset<DESC_LEN> >& descriptors, const vector<cv::KeyPoint>& kpts, IplImage* img);

	private:
		void pickTestLocations(void);

		void allocateIntegralImage(const IplImage* img);
		bool validateKeypoints(const vector< cv::KeyPoint >& kpts, int im_w, int im_h);
		bool isKeypointInsideSubImage(const cv::KeyPoint& kpt, const int width, const int height);
		vector< pair<cv::Point2i, cv::Point2i> > testLocations;

		CvMat* integralImage;
	};

};

#endif
