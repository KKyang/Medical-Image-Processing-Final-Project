#include "breastmassspiculatiodetect.h"
#include <QDebug>
BreastMassSpiculatioDetect::BreastMassSpiculatioDetect(QObject *parent) : QObject(parent)
{
    srand(time(NULL));
}

//find histogram
void BreastMassSpiculatioDetect::histogram(cv::Mat &inputArray, cv::Mat &hist)
{
    std::vector<std::vector<int>> data;
    histogram(inputArray, hist, data);
}

//find histogram. Also return data results.
void BreastMassSpiculatioDetect::histogram(cv::Mat &inputArray, cv::Mat &hist, std::vector<std::vector<int>> &data)
{
    int i=0;
    int largestCount = 0;
    //type check
    if(inputArray.type() == CV_8UC3) //color
    {
        cv::Mat&& result = cv::Mat(100, 256, CV_8UC3, cv::Scalar(255,255,255));
        data.resize(3); //resize vector size to 3 channels
        data[0].resize(256, 0); //each 256 blocks
        data[1].resize(256, 0);
        data[2].resize(256, 0);
        //Calculate histogram
        #pragma omp parallel for private(i)
        for(int j=0; j < inputArray.size().height;j++)
        {
            for(i = 0; i < inputArray.size().width;i++)
            {
                data[0][inputArray.at<cv::Vec3b>(j,i)[0]]++;
                data[1][inputArray.at<cv::Vec3b>(j,i)[1]]++;
                data[2][inputArray.at<cv::Vec3b>(j,i)[2]]++;
            }
        }
        //Normalization
        for (i = 0; i < 256; i++)
        {
            for(int k = 0; k < 3; k++)
            {
                if (data[k][i] > largestCount)
                {
                    largestCount = data[k][i];
                }
            }
        }
        //Draw histogram results to mat.
        int j = 0; int k = 0;
        #pragma omp parallel for private(j, k)
        for (i = 0; i < 256; i ++)
        {
            for(k = 0; k < 3 ; k++)
            {
                for (j = 0; j < 100-(int)(((double)data[k][i] / (double)largestCount) * 100); j++)
                {
                    result.at<cv::Vec3b>(j,i)[k] = 0;
                }
            }
        }
        hist.release();
        hist = result.clone();
        result.release();
    }
    else if(inputArray.type() == CV_8UC1) //gray
    {
        cv::Mat&& result = cv::Mat(100, 256, CV_8UC1, cv::Scalar(255,255,255));
        data.resize(1); //resize vector
        data[0].resize(256, 0);
        //Calculate histogram/
        #pragma omp parallel for private(i)
        for(int j=0; j < inputArray.size().height;j++)
        {
            for(i = 0; i < inputArray.size().width;i++)
            {
                data[0][inputArray.at<uchar>(j,i)]++;
            }
        }
        //Normalization
        for (i = 0; i < 256; i++)
        {
            if (data[0][i] > largestCount)
            {
                largestCount = data[0][i];
            }
        }
        //Draw histogram
        int j = 0;
        #pragma omp parallel for private(j)
        for (i = 0; i < 256; i ++)
        {
            for (j = 0; j < 100-(int)(((double)data[0][i] / (double)largestCount) * 100); j++)
            {
                    result.at<uchar>(j,i) = 0;
            }
        }
        //Output as BGR format. (Black-n-white image)
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
        hist.release();
        hist = result.clone();
        result.release();
    }
}

void BreastMassSpiculatioDetect::setImage(cv::Mat &img)
{
    cv::Mat tmp;
    cv::cvtColor(img, tmp, cv::COLOR_BGR2GRAY);
    double left = 0, right = 0;
    for(int j; j < tmp.rows; j++)
    {
        for(int i = 0; i < tmp.cols; i++)
        {
            if(i < tmp.cols / 2)
            {
                left += (double)tmp.at<uchar>(j,i);
            }
            else
            {
                right += (double)tmp.at<uchar>(j,i);
            }
        }
    }

    if(left > right)
    {
        flip = true;
        cv::flip(img, _orignal_img, 1);
    }
    else
    {
        flip = false;
        _orignal_img = img.clone();
    }
    qDebug() << left << " " << right << " " << flip;

}

void BreastMassSpiculatioDetect::HAARTransform()
{
    //OpenCV HAAR Transform. Available at: http://answers.opencv.org/question/42273/wavelet-transform/
    //I use HAAR Transform to replace the frequency pyramid.
    cv::Mat im,im1,im2,im3,im4,im5,im6, imi;
    float a,b,c,d;

    im = _orignal_img.clone();
    cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
    imi=cv::Mat::zeros(im.rows,im.cols,CV_8U);
    im.copyTo(imi);
    im.convertTo(im,CV_32F,1.0,0.0);
    im1=cv::Mat::zeros(im.rows/2,im.cols,CV_32F);
    im2=cv::Mat::zeros(im.rows/2,im.cols,CV_32F);
    im3=cv::Mat::zeros(im.rows/2,im.cols/2,CV_32F);
    im4=cv::Mat::zeros(im.rows/2,im.cols/2,CV_32F);
    im5=cv::Mat::zeros(im.rows/2,im.cols/2,CV_32F);
    im6=cv::Mat::zeros(im.rows/2,im.cols/2,CV_32F);


    //--------------Decomposition-------------------

    for(int rcnt=0;rcnt<im.rows;rcnt+=2)
    {
        for(int ccnt=0;ccnt<im.cols;ccnt++)
        {
            a=im.at<float>(rcnt,ccnt);
            b=im.at<float>(rcnt+1,ccnt);
            c=(a+b)*0.707;
            d=(a-b)*0.707;
            int _rcnt=rcnt/2;
            im1.at<float>(_rcnt,ccnt)=c;
            im2.at<float>(_rcnt,ccnt)=d;
        }
    }

    for(int rcnt=0;rcnt<im.rows/2;rcnt++)
    {
        for(int ccnt=0;ccnt<im.cols;ccnt+=2)
        {
            a=im1.at<float>(rcnt,ccnt);
            b=im1.at<float>(rcnt,ccnt+1);
            c=(a+b)*0.707;
            d=(a-b)*0.707;
            int _ccnt=ccnt/2;
            im3.at<float>(rcnt,_ccnt)=c;
            im4.at<float>(rcnt,_ccnt)=d;
        }
    }

    for(int rcnt=0;rcnt<im.rows/2;rcnt++)
    {
        for(int ccnt=0;ccnt<im.cols;ccnt+=2)
        {
            a=im2.at<float>(rcnt,ccnt);
            b=im2.at<float>(rcnt,ccnt+1);
            c=(a+b)*0.707;
            d=(a-b)*0.707;
            int _ccnt=ccnt/2;
            im5.at<float>(rcnt,_ccnt)=c;
            im6.at<float>(rcnt,_ccnt)=d;
        }
    }

    cv::normalize(im3,im3, 255.0, 0.0, cv::NORM_MINMAX);
    im3.convertTo(_haar_img, CV_8U);
    //cv::imshow("downgraded1", _haar_img);
    sendResultImg(_haar_img, 0);
}

void BreastMassSpiculatioDetect::ROI()
{
    //OAT
    int hist_c0[256] = {}, hist_c1[256] = {}; //c0 - background, c1- object
    double max, min;
    double Gmax = 0;
    int threshold_value = 255, last_threshold_value = 255;


    double A = 0;
    for(int j = 0; j < _haar_img.rows; j++)
        for(int i = 0; i < _haar_img.cols; i++)
        {
            if(_haar_img.at<uchar>(j,i) > Gmax) Gmax = _haar_img.at<uchar>(j,i);
            A += _haar_img.at<uchar>(j,i);
        }
    A /= (double)(_haar_img.rows * _haar_img.cols);

    while(1)
    {

        int TH = 0, GH = 0;
        for(int j = 0; j < _haar_img.rows; j++)
            for(int i = 0; i < _haar_img.cols; i++)
            {
                if(_haar_img.at<uchar>(j,i) <= threshold_value) TH++ ;
                if(_haar_img.at<uchar>(j,i) <= Gmax) GH++;
            }
        qDebug() << "threshold value " << threshold_value << " A " << A << " TH, GH " << TH << ", " << OTSU_beta * GH;
        if(threshold_value <= A || TH <= OTSU_beta * (double)GH)
            break;

        //calculate histogram
        for(int j = 0; j < _haar_img.rows; j++)
            for(int i = 0; i < _haar_img.cols; i++)
            {
                if(_haar_img.at<uchar>(j,i) <= threshold_value){hist_c0[_haar_img.at<uchar>(j,i)]++;}
            }
        //OTSU
        double&& sum = 0;
        double&& weight_sum = 0;
        for(int j = 0; j < 256; j++)
        {
                sum += hist_c0[j];
                weight_sum += (j + 1) * hist_c0[j];
        }

        double a, w_a, u, v, g[256];
        for(int j = 0; j < 256; j++)
        {
            a=w_a=0;
            for(int i = 0; i <= j; i++)
            {
                a+= hist_c0[i];
                w_a+= hist_c0[i] * (i + 1);
            }
            if(a){u = w_a / a;}
            else u = 0;
            if(sum - a){v = (weight_sum-w_a)/(sum-a);}
            else v = 0;
            g[j]= a *(sum - a)*(u - v) * (u - v);
        }

        max = min = g[0];

        for(int j = 0; j < 256; j++)
        {
            if(g[j] > max)
            {
                max = g[j];
                threshold_value = j;
            }
            else if(g[j] < min){min = g[j];}
        }

        if(threshold_value > last_threshold_value)
        {
            threshold_value = last_threshold_value;
            break;
        }

        last_threshold_value = threshold_value;
    }
    cv::Mat OATMAP;
    cv::threshold(_haar_img, OATMAP, threshold_value, 255, cv::THRESH_BINARY);
    _otsu_img = OATMAP.clone();
    //cv::imshow("result", OATMAP);
    sendResultImg(_otsu_img, 1);


    //hill climbing & region growing
    std::vector<std::vector<std::vector<int>>> hill_result;
    cv::Mat tmp = cv::Mat(1, _haar_img.cols, CV_8UC1).clone();
    tmp.create(cv::Size(1,1), CV_8UC3);


    for(int rows = 0; rows < _haar_img.rows; rows++)
    {
        _haar_img.row(rows).copyTo(tmp, OATMAP.row(rows));
        std::vector<std::vector<int>> path;
        path.push_back(std::vector<int>());
        path[0].push_back(_haar_img.at<uchar>(rows, 0));

        for(int i = 1; i < _haar_img.cols - 1;i++)
        {
            int value_l = _haar_img.at<uchar>(rows, i - 1);
            int value   = _haar_img.at<uchar>(rows, i);
            int value_r = _haar_img.at<uchar>(rows, i + 1);
            path[path.size() - 1].push_back(value);
            while(value_r == value)
            {
                i++;
                value_r = _haar_img.at<uchar>(rows, i + 1);
                path[path.size() - 1].push_back(value);
            }
            if(value < value_l && value < value_r)
            {
                path.push_back(std::vector<int>());
            }
        }
        path[path.size() - 1].push_back(_haar_img.at<uchar>(rows, _haar_img.cols - 1));
        hill_result.push_back(path);
    }

    cv::Mat hill_img(_haar_img.size(), CV_8UC1);
    for(int j = 0; j < hill_img.rows; j++)
    {
        for(int i = 0; i < hill_img.cols; i++)
        {
           hill_img.at<uchar>(j, i) = 255;
        }
    }
    int last_cut_off_point = hill_img.rows / 2;
    for(int rows = 0; rows < hill_img.rows; rows++)
    {
        std::vector<std::vector<int>> &data = hill_result[rows];
        int cut_off_point = data[data.size() - 1].size();
        for(int i = data.size() - 2; i >= 1; i--)
        {
            auto&& value_l = data[i - 1].size();
            auto&& value = data[i].size();
            double avg_l = 0, avg = 0;
            for(int a = 0; a < value_l; a++)
            {
                avg_l += data[i - 1][a];
            }
            avg_l /= value_l;
            for(int a = 0; a < value; a++)
            {
                avg += data[i][a];
            }
            avg /= value;

            cut_off_point += data[i].size();
//            if(avg < 140 && avg_l < 140)
//            {
//                break;
//            }
            if(cut_off_point > last_cut_off_point * 1.1)
            {
                cut_off_point -= data[i].size();
                break;
            }
            //qDebug () << cut_off_point << " " << last_cut_off_point;

            if(abs(avg - avg_l) > 5 && (avg < 160))// && avg > 140))
            {
                int cut_count = cut_off_point;
                int pos = i - 1;
                bool if_break= false;
                int count = 0;
                int count2 = 0;
                while(pos >= 1)
                {
                    if(cut_count > last_cut_off_point * 1.1)
                    {
                        if_break = true;
                        break;
                    }
                    double avg_ll;
                    for(int a = 0; a < data[pos].size(); a++)
                    {
                        avg_ll += data[pos][a];
                    }
                    avg_ll = avg_ll / data[pos].size();
                    cut_count += data[pos].size();
                    if(abs(avg - avg_ll) > 5 && (avg_ll < 160))// && avg_ll > 140))
                    {
                        count++;
                        count2++;
                    }
                    else
                    {
                        count--;
                    }
                    if(count == 10)
                    {
                        if_break = true;
                        break;
                    }
                    if(count2 == (hill_img.cols / 5))
                    {
                        break;
                    }


                    pos--;
                }
                if(if_break)
                    break;
            }

        }
        for(int i = hill_img.cols - 1; i >= hill_img.cols - 1 - cut_off_point ; i--)
        {
            hill_img.at<uchar>(rows, i) = 0;
        }
        last_cut_off_point = cut_off_point;

    }
    //cv::imshow("hill result", hill_img);
    _haar_img.copyTo(_hill_img, hill_img);
    _hill_img.copyTo(_hill_img, OATMAP);

    sendResultImg(_hill_img, 2);
    cv::Mat haar_no_muscle = _hill_img.clone();

    //Breast ratio calculation
    int bmax = 0;

    sublinetop = 0, sublinebut = 0, sublineleft = 0, sublineright = 0;
    for(int j = 0; j < haar_no_muscle.rows; j++)
    {
        int count = 0;
        for(int i = 0; i < haar_no_muscle.cols; i++)
        {
            if(OATMAP.at<uchar>(j, i) == 255 && hill_img.at<uchar>(j, i) == 255)
            {
                count++;
            }
        }
        if(count > bmax){bmax = count;}
    }
    qDebug() << bmax;

    for(int j = 0; j < haar_no_muscle.rows; j++)
    {
        int count = 0;
        for(int i = 0; i < haar_no_muscle.cols; i++)
        {
            if(OATMAP.at<uchar>(j, i) == 255 && hill_img.at<uchar>(j, i) == 255)
            {
                count++;
            }
        }

        if(count >= (double)bmax * breast_ratio_x && j > haar_no_muscle.rows / 8)
        {
            sublinetop = j;
            break;
        }
    }

    for(int j = haar_no_muscle.rows - 1; j >= 0; j--)
    {
        int count = 0;
        for(int i = 0; i < haar_no_muscle.cols; i++)
        {
            if(OATMAP.at<uchar>(j, i) == 255 && hill_img.at<uchar>(j, i) == 255)
            {
                count++;
            }
        }
        if(count >= (double)bmax * breast_ratio_x)
        {
            sublinebut = j;
            break;
        }
    }
    haar_no_muscle = haar_no_muscle(cv::Rect(0, sublinetop, haar_no_muscle.cols, sublinebut - sublinetop));
    cv::Mat cut_OAT  = OATMAP(cv::Rect(0, sublinetop, haar_no_muscle.cols, sublinebut - sublinetop));
    cv::Mat cut_hill = hill_img(cv::Rect(0, sublinetop, haar_no_muscle.cols, sublinebut - sublinetop));
    for(int i = 0; i < haar_no_muscle.cols; i++)
    {
        int count = 0;
        for(int j = 0; j < haar_no_muscle.rows; j++)
        {
            if(cut_OAT.at<uchar>(j, i) == 255 && cut_hill.at<uchar>(j, i) == 255)
            {
                count++;
            }
        }
        if(count >= 50)
        {
            sublineleft = i - 10;
            if(sublineleft < 0){sublineleft = 0;}
            break;
        }
    }
    haar_no_muscle = haar_no_muscle(cv::Rect(sublineleft, 0, haar_no_muscle.cols - sublineleft, haar_no_muscle.rows));
    cut_OAT  = OATMAP(cv::Rect(sublineleft, 0, haar_no_muscle.cols - sublineleft, haar_no_muscle.rows));
    cut_hill = hill_img(cv::Rect(sublineleft, 0, haar_no_muscle.cols - sublineleft, haar_no_muscle.rows));

    for(int i = haar_no_muscle.cols - 1; i >= 0; i--)
    {
        int count = 0;
        for(int j = 0; j < haar_no_muscle.rows; j++)
        {
            if(cut_OAT.at<uchar>(j, i) == 255 && cut_hill.at<uchar>(j, i) == 255)
            {
                count++;
            }
        }
        if(count >= haar_no_muscle.rows * 0.1)
        {
            sublineright = i;
            break;
        }
    }
    qDebug() << "right: " << sublineright;

    _ROI_img = haar_no_muscle(cv::Rect(0, 0, sublineright, haar_no_muscle.rows));

    //cv::imshow("ROI", _ROI_img);
    sendResultImg(_ROI_img, 3);
}

void BreastMassSpiculatioDetect::CLACHE()
{
    cv::Ptr<cv::CLAHE> p = cv::createCLAHE();
    p->setClipLimit(clipLimit);
    p->setTilesGridSize(gridSize);
    p->apply(_ROI_img, _CLACHEROI_img);
    //cv::imshow("CLACHEROI", _CLACHEROI_img);
    sendResultImg(_CLACHEROI_img, 4);
}

void BreastMassSpiculatioDetect::fuzzyClustering()
{
    cv::Mat tmp;
    if(_CLACHEROI_img.type() != CV_8UC1)
    {
        cv::cvtColor(_CLACHEROI_img, tmp, cv::COLOR_BGR2GRAY);
    }
    else
    {
        tmp = _CLACHEROI_img.clone();
    }

    cv::Mat hist;

    std::vector<std::vector<int>> data;
    histogram(tmp, hist, data);
    //cv::imshow("hist", hist);


    int new_pos = 1;
    cv::Mat cluster_ROI(_CLACHEROI_img.size(),CV_32FC1);
    cluster_ROI = cv::Scalar(-1);
    for(int j = 0; j < cluster_ROI.rows; j++)
    {
        for(int i = 0; i < cluster_ROI.cols; i++)
        {
            if(_ROI_img.at<uchar>(j, i) == 0)
            {
                cluster_ROI.at<float>(j, i) = 0;
            }
        }
    }

    int iter = 100;
    int count = 0;
    while(count < iter)
    {
        cv::Mat dataset(cv::Size(2, 256 - new_pos), CV_32FC1);
        for(int j = 0; j < dataset.rows; j++)
        {
            dataset.at<float>(j, 0) = (float)j + new_pos;
            dataset.at<float>(j, 1) = (float)data[0][j];
        }
        SoftC::Fuzzy f (dataset, number_clusters, fuzziness, epsilon, dist_type, init_type);
        f.clustering (num_iterations);
        cv::Mat centroids = f.get_centroids_ ();
        if(abs(centroids.at<float>(0, 0) - centroids.at<float>(1, 0)) < 3)
            break;
        new_pos = cv::saturate_cast<uchar>((int)(centroids.at<float>(0, 0)) - abs(centroids.at<float>(0, 0) - centroids.at<float>(1, 0))/2);


        for(int j = 0; j < cluster_ROI.rows; j++)
        {
            for(int i = 0; i < cluster_ROI.cols; i++)
            {
                if(cluster_ROI.at<float>(j, i) == -1)
                {
                    if(_CLACHEROI_img.at<uchar>(j, i) <= new_pos)
                    {
                        cluster_ROI.at<float>(j, i) = centroids.at<float>(1, 0);
                    }
                }
            }
        }
        count++;
    }

    for(int j = 0; j < cluster_ROI.rows; j++)
    {
        for(int i = 0; i < cluster_ROI.cols; i++)
        {
            if(cluster_ROI.at<float>(j, i) == -1)
            {
                cluster_ROI.at<float>(j, i) = 400;

            }
        }
    }


    cv::normalize(cluster_ROI, cluster_ROI, 0, 255, CV_MINMAX);
    cluster_ROI.convertTo(cluster_ROI, CV_8U);
    _fuzzy_img = cluster_ROI.clone();
    sendResultImg(_fuzzy_img, 5);

}

int BreastMassSpiculatioDetect::acTiveContourModel()
{
    cv::namedWindow("clustered roi");

    cv::Mat fuzzy_color;
    cv::cvtColor(_fuzzy_img, fuzzy_color, cv::COLOR_GRAY2BGR);
    cv::Mat tmp = fuzzy_color.clone();
    std::vector<std::vector<cv::Point>> point_cv(1);
    cv::setMouseCallback("clustered roi", CallBackFunc, &point_cv[0]);

    while(!stopped)
    {
        cv::imshow("clustered roi", tmp);
        if(point_cv[0].size() > 1)
        {
            tmp = fuzzy_color.clone();
            cv::polylines(tmp, point_cv, true,cv::Scalar(0,0,255), 4);
        }
        char key = cv::waitKey(10);
        key = std::tolower(key);

        if('a' == key)
        {
            break;
        }
        else if('b' == key)
        {
            tmp = fuzzy_color.clone();
            point_cv.clear();
            point_cv.resize(1);
        }
        else if('r' == key)
        {
            if(point_cv[0].size())
                point_cv[0].erase(point_cv[0].end() - 1);
        }
        else if('x' == key)
        {
            cv::destroyAllWindows();
            return 0;
        }
    }
    if(stopped)
    {
        stopped = false;
        cv::destroyAllWindows();
        return 0;
    }
    cv::destroyAllWindows();
    _initial_boundary_img = tmp.clone();
    int length = point_cv[0].size();
    CvPoint* point = new CvPoint[length];
    for(int i = 0; i < length; i++)
    {
        point[i].x = point_cv[0][i].x;
        point[i].y = point_cv[0][i].y;
    }

    CvMat ACM = _fuzzy_img;
    point = cvSnakeImageGVF(&ACM, point, &length, alpha, beta, gamma, kappa, 50, 10, CV_REINITIAL, CV_GVF);
    point_cv.clear();
    point_cv.resize(1);

    double ratioH = (double)_orignal_img.cols / _haar_img.cols;
    for(int i = 0; i < length; i++)
    {
        point_cv[0].push_back(cv::Point((point[i].x + sublineleft) * ratioH, (point[i].y + sublinetop) * ratioH));
    }


//    haar_no_muscle = haar_no_muscle(cv::Rect(0, sublinetop, haar_no_muscle.cols, sublinebut - sublinetop));
//    haar_no_muscle = haar_no_muscle(cv::Rect(sublineleft, 0, haar_no_muscle.cols - sublineleft, haar_no_muscle.rows));
//    _ROI_img = haar_no_muscle(cv::Rect(0, 0, sublineright, haar_no_muscle.rows));

    tmp = _orignal_img.clone();
    cv::polylines(tmp, point_cv, true,cv::Scalar(0,0,255), 4);
    //cv::imshow("final result", tmp);
    delete point;
    _final_result = tmp.clone();
    sendResultImg(_final_result, 6);
    sendResultImg(_initial_boundary_img, 7);
    return 1;
}

void BreastMassSpiculatioDetect::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == cv::EVENT_LBUTTONDOWN )
     {
         std::vector<cv::Point>* points = (std::vector<cv::Point>*)userdata;
         points->push_back(cv::Point(x, y));
         std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;

     }
//     else if ( event == cv::EVENT_MOUSEMOVE )
//     {
//          std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;

//     }
}



//Wavelet transform
//    imd=cv::Mat::zeros(im.rows,im.cols,CV_32F);

//    im3.copyTo(imd(cv::Rect(0,0,128,128)));
//    im4.copyTo(imd(cv::Rect(0,127,128,128)));
//    im5.copyTo(imd(cv::Rect(127,0,128,128)));
//    im6.copyTo(imd(cv::Rect(127,127,128,128)));


    //---------------------------------Reconstruction-------------------------------------

//    imr=cv::Mat::zeros(im.rows,im.cols,CV_32F);
//    im11=cv::Mat::zeros(im.rows/2,im.cols,CV_32F);
//    im12=cv::Mat::zeros(im.rows/2,im.cols,CV_32F);
//    im13=cv::Mat::zeros(im.rows/2,im.cols,CV_32F);
//    im14=cv::Mat::zeros(im.rows/2,im.cols,CV_32F);

//    for(int rcnt=0;rcnt<im.rows/2;rcnt++)
//    {
//        for(int ccnt=0;ccnt<im.cols/2;ccnt++)
//        {
//            int _ccnt=ccnt*2;
//            im11.at<float>(rcnt,_ccnt)=im3.at<float>(rcnt,ccnt);     //Upsampling of stage I
//            im12.at<float>(rcnt,_ccnt)=im4.at<float>(rcnt,ccnt);
//            im13.at<float>(rcnt,_ccnt)=im5.at<float>(rcnt,ccnt);
//            im14.at<float>(rcnt,_ccnt)=im6.at<float>(rcnt,ccnt);
//        }
//    }


//    for(int rcnt=0;rcnt<im.rows/2;rcnt++)
//    {
//        for(int ccnt=0;ccnt<im.cols;ccnt+=2)
//        {

//            a=im11.at<float>(rcnt,ccnt);
//            b=im12.at<float>(rcnt,ccnt);
//            c=(a+b)*0.707;
//            im11.at<float>(rcnt,ccnt)=c;
//            d=(a-b)*0.707;                           //Filtering at Stage I
//            im11.at<float>(rcnt,ccnt+1)=d;
//            a=im13.at<float>(rcnt,ccnt);
//            b=im14.at<float>(rcnt,ccnt);
//            c=(a+b)*0.707;
//            im13.at<float>(rcnt,ccnt)=c;
//            d=(a-b)*0.707;
//            im13.at<float>(rcnt,ccnt+1)=d;
//        }
//    }

//    temp=cv::Mat::zeros(im.rows,im.cols,CV_32F);

//    for(int rcnt=0;rcnt<im.rows/2;rcnt++)
//    {
//        for(int ccnt=0;ccnt<im.cols;ccnt++)
//        {

//            int _rcnt=rcnt*2;
//            imr.at<float>(_rcnt,ccnt)=im11.at<float>(rcnt,ccnt);     //Upsampling at stage II
//            temp.at<float>(_rcnt,ccnt)=im13.at<float>(rcnt,ccnt);
//        }
//    }

//    for(int rcnt=0;rcnt<im.rows;rcnt+=2)
//    {
//        for(int ccnt=0;ccnt<im.cols;ccnt++)
//        {

//            a=imr.at<float>(rcnt,ccnt);
//            b=temp.at<float>(rcnt,ccnt);
//            c=(a+b)*0.707;
//            imr.at<float>(rcnt,ccnt)=c;                                      //Filtering at Stage II
//            d=(a-b)*0.707;
//            imr.at<float>(rcnt+1,ccnt)=d;
//        }
//    }

//Old hill-climbing code
//std::vector<int> climb_rows;
//std::vector<cv::Point> climb_points;
//for(int j = 0; j < _haar_img.rows; j++)
//{
//    int step = 1;
//    int rate = 1;
//    int i = 0, z = 0;
//    cv::Point p;

//    i = rand() % _haar_img.cols;

//    while(step)
//    {
//        if(OATMAP.at<uchar>(j, i) > 0)
//        {
//            auto&& z2 = _haar_img.at<uchar>(j, i);
//            if(z2 > z){z = z2; p = cv::Point(i,j);}
//            //else {step -= rate;}
//        }
//        i = i + step;
//        if(i >= _haar_img.cols){ i = _haar_img.cols - 1; break;}
//    }
//    if(z >= threshold_value)
//    {
//        climb_rows.push_back(z);
//        climb_points.push_back(p);
//    }
//}

//qDebug() << "rows " << _haar_img.rows << " hill " << climb_rows.size();


//cv::Mat no_muscle = cv::Mat::zeros(_haar_img.size(), CV_8UC1);
//std::sort(climb_rows.begin(), climb_rows.end());
////qDebug() << climb_rows[climb_rows.size() - 1];
//for(int j = 0; j < _haar_img.rows; j++)
//{
//    for(int i = 0; i < _haar_img.cols; i++)
//    {
//        if(!(_haar_img.at<uchar>(j, i) < climb_rows[climb_rows.size() - 1] + 30 && _haar_img.at<uchar>(j, i) > climb_rows[climb_rows.size() - 1] - 100))
//        {
//            no_muscle.at<uchar>(j, i) = 255;
//        }
//    }
//}
//cv::imshow("no_muscle", no_muscle);
