#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/gpumat.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

void detectAndDisplay(Mat frame);

CascadeClassifier faceCascade;
char * windowName = "CUDA 6 Optimized Facial Tracking In OpenCV";
RNG rng(12345);

int main(int argc, char * argv[])
{
	VideoCapture capture;
	Mat frame;
	//use the haarcascade_frontalface_alt.xml library
	cout << "Load? ";
	cout << faceCascade.load("haarcascade_frontalface_default.xml") << endl;
	capture.open(0);

	//namedWindow(windowName, 1);
	while (capture.isOpened()) 
	{
		capture >> frame;
		if (!frame.empty()) {
			detectAndDisplay(frame);
		} else {
			printf("--! Error No captured frame\n");
			break;
		}
		int c = waitKey(30);
		if ((char)c == 'q') { break; }
	}
	return 0;
}

void detectAndDisplay(Mat frame) {
	Mat gray;
	cvtColor(frame, gray, CV_BGR2GRAY); //grayscale the frame into new object
	equalizeHist(gray, gray); //equalize and store in the same object

	std::vector<Rect> faces;
	//store faces into vector 
	faceCascade.detectMultiScale(gray, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	cout<<faces.size()<<" ";
	for (int i = 0; i < faces.size(); i++) {
		Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		Point pt2(faces[i].x, faces[i].y);

		rectangle(frame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
	}
		 
	imshow(windowName, frame);
}