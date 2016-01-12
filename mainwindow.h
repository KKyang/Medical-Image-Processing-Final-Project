#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtConcurrent/QtConcurrent>
#include <QMessageBox>
#include <QProgressBar>
#include <QFileDialog>
#include <QTimer>
#include "breastmassspiculatiodetect.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void receiveResultImg(const cv::Mat &img, int num);
    void GVFFinished();
    void updateProgressBar();
    void radioButtonChanged(bool s);
    void on_actionOpen_image_triggered();

    void on_pushButton_process_clicked();

    void on_actionRead_Groundtruth_triggered();

private:
    Ui::MainWindow *ui;
    QProgressBar *progressBar;

    BreastMassSpiculatioDetect BMSD;

    cv::Mat _opened_img;
    cv::Mat _haar_img;
    cv::Mat _otsu_img;
    cv::Mat _hill_img;
    cv::Mat _ROI_img;
    cv::Mat _CLACHEROI_img;
    cv::Mat _fuzzy_img;
    cv::Mat _final_result;
    cv::Mat _initial_boundary_img;

    cv::Mat _groundTruth;

    QFuture<int> _gvf_future;
    QFutureWatcher<int> _gvf_watcher;

    QTimer count;
};

#endif // MAINWINDOW_H
