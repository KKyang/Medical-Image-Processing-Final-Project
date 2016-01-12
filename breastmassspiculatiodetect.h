#ifndef BREASTMASSSPICULATIODETECT_H
#define BREASTMASSSPICULATIODETECT_H

#include <QObject>
#include <time.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "fuzzy_clustering.hpp"
#include "activeContour/gvfc.h"

class BreastMassSpiculatioDetect : public QObject
{
    Q_OBJECT
public:
    explicit BreastMassSpiculatioDetect(QObject *parent = 0);
    void setImage(cv::Mat &img);
    void HAARTransform();
    void ROI();
    void CLACHE();
    void fuzzyClustering();
    int acTiveContourModel();
    void stopWindow(){stopped = true;}

    const bool isFlipped(){return flip;}

    //        Set Params        //
    void setOTSUBeta(double b = 0.735){OTSU_beta = b;}
    void setBreastRatio(double r = 0.62){breast_ratio_x = r;}

    void setCLACHEClipLimit(double cl){clipLimit = cl;}
    void setCLACHEGridSize(cv::Size gs){gridSize = gs;}
    void setFuzzyClusterNum(unsigned int nc = 2){number_clusters = nc;}
    void setFuzzyFuzziness(float f = 2.0){fuzziness = f;}
    void setFuzzyEpsilon(float eps = 0.01){epsilon = eps;}
    void setFuzzyIteration(unsigned int iter = 100){num_iterations = iter;}
    void setFuzzyDistType(SoftCDistType type = kSoftCDistL2){dist_type = type;}
    void setFuzzyInitType(SoftCInitType type = kSoftCInitKmeansPP){init_type = type;}

    void setSnakeParams(float a = 0.05, float b = 0.1, float g = 1.0, float k = 2.0){alpha = a, beta = b, gamma = g, kappa = k;}

signals:
    void sendResultImg(const cv::Mat &img, int num);

public slots:
private:
    void histogram(cv::Mat &inputArray, cv::Mat &hist);
    void histogram(cv::Mat &inputArray, cv::Mat &hist, std::vector<std::vector<int>> &data);
    static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
    bool flip = false;
    cv::Mat _orignal_img;
    cv::Mat _haar_img;
    cv::Mat _otsu_img;
    cv::Mat _hill_img;
    cv::Mat _ROI_img;
    cv::Mat _CLACHEROI_img;
    cv::Mat _fuzzy_img;
    cv::Mat _initial_boundary_img;
    cv::Mat _final_result;

    //        Params        //
    double OTSU_beta = 0.735; //Recursive OTSU manual treshold.
    double breast_ratio_x = 0.62; //Breast ratio
    int sublinetop = 0, sublinebut = 0, sublineleft = 0, sublineright = 0;

    //CLACHE
    cv::Size gridSize = cv::Size(8, 8);
    double clipLimit = 40;
    //Fuzzy Clustering
    unsigned int number_clusters = 2;    // クラスタ数
    float fuzziness = 2.0;                      // 乱雑さ
    float epsilon = 0.01;                       // 終了閾値
    unsigned int num_iterations = 100;          // 繰り返しの回数
    SoftCDistType dist_type = kSoftCDistL2;         // 距離指標
    SoftCInitType init_type = kSoftCInitKmeansPP;   // 初期化方法

    //Snake GVF params
    bool stopped = false;
    float alpha=0.05f, beta=0.1f, gamma=1.0f, kappa=2.0f;

};

#endif // BREASTMASSSPICULATIODETECT_H
