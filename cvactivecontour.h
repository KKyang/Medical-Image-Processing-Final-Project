#ifndef CVACTIVECONTOUR_H
#define CVACTIVECONTOUR_H
#define  CV_VALUE  1
#define  CV_ARRAY  2
#include "opencv2/opencv.hpp"
/* Updates active contour in order to minimize its cummulative
   (internal and external) energy. */
void  cvSnakeImage( const IplImage* image, CvPoint* points,
                    int  length, float* alpha,
                    float* beta, float* gamma,
                    int coeff_usage, CvSize  win,
                    CvTermCriteria criteria, int calc_gradient CV_DEFAULT(1));

#endif // CVACTIVECONTOUR_H
