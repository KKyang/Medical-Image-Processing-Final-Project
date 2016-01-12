#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->radioButton_resultImg->setEnabled(false);
    ui->radioButton_initialBoundary->setEnabled(false);
    connect(&BMSD, SIGNAL(sendResultImg(cv::Mat,int)), this, SLOT(receiveResultImg(cv::Mat,int)));
    connect(this->ui->radioButton_originImg, SIGNAL(toggled(bool)), this, SLOT(radioButtonChanged(bool)));
    connect(this->ui->radioButton__haarImg, SIGNAL(toggled(bool)), this, SLOT(radioButtonChanged(bool)));
    connect(this->ui->radioButton_ROIImg, SIGNAL(toggled(bool)), this, SLOT(radioButtonChanged(bool)));
    connect(this->ui->radioButton_CLACHEImg, SIGNAL(toggled(bool)), this, SLOT(radioButtonChanged(bool)));
    connect(this->ui->radioButton_fuzzyImg, SIGNAL(toggled(bool)), this, SLOT(radioButtonChanged(bool)));
    connect(this->ui->radioButton_resultImg, SIGNAL(toggled(bool)), this, SLOT(radioButtonChanged(bool)));
    connect(this->ui->radioButton_initialBoundary, SIGNAL(toggled(bool)), this, SLOT(radioButtonChanged(bool)));
    connect(&_gvf_watcher, SIGNAL(finished()), this, SLOT(GVFFinished()));
    connect(&count, SIGNAL(timeout()), this, SLOT(updateProgressBar()));

    progressBar = new QProgressBar(NULL);
    progressBar->setMaximumHeight(16);
    progressBar->setMaximumWidth(200);
    progressBar->setTextVisible(true);
    progressBar->setAlignment(Qt::AlignRight);
    count.setInterval(100);
    this->statusBar()->addPermanentWidget(progressBar);
}

MainWindow::~MainWindow()
{
    progressBar->deleteLater();
    if(_gvf_watcher.isRunning())
    {
        BMSD.stopWindow();
        BMSD.deleteLater();
    }

    delete ui;
}

void MainWindow::receiveResultImg(const cv::Mat &img, int num)
{
    cv::Mat tmp;
    if(BMSD.isFlipped())
    {
        cv::flip(img, tmp, 1);
    }
    else
    {
        tmp = img.clone();
    }
    switch(num)
    {
    case 0:
        _haar_img = tmp.clone();
        break;
    case 1:
        _otsu_img = tmp.clone();
        break;
    case 2:
        _hill_img = tmp.clone();
        break;
    case 3:
        _ROI_img = tmp.clone();
        break;
    case 4:
        _CLACHEROI_img = tmp.clone();
        break;
    case 5:
        _fuzzy_img = tmp.clone();
        break;
    case 6:
        _final_result = tmp.clone();
        break;
    case 7:
        _initial_boundary_img = tmp.clone();
        break;
    }
}

void MainWindow::radioButtonChanged(bool s)
{
    std::vector<cv::Mat> tmp;
    if(ui->radioButton_originImg->isChecked())
    {
        tmp.push_back(_opened_img);
    }
    else if(ui->radioButton__haarImg->isChecked())
    {
        tmp.push_back(_haar_img);
    }
    else if(ui->radioButton_ROIImg->isChecked())
    {
        tmp.push_back(_otsu_img);
        tmp.push_back(_hill_img);
        tmp.push_back(_ROI_img);
    }
    else if(ui->radioButton_CLACHEImg->isChecked())
    {
        tmp.push_back(_CLACHEROI_img);
    }
    else if(ui->radioButton_fuzzyImg->isChecked())
    {
        tmp.push_back(_fuzzy_img);
    }
    else if(ui->radioButton_resultImg->isChecked())
    {
        if(!_groundTruth.empty())
        {
            if(_groundTruth.size() != _final_result.size())
            {
                _groundTruth.release();
                QMessageBox::warning(this, "Error", "Size not match, releasing ground truth data.");
                return;
            }
            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(_groundTruth, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

            for( int i = 0; i< contours.size(); i++ )
            {
              drawContours(_final_result, contours, i, cv::Scalar(0, 255, 0), 2, 8, hierarchy, 0, cv::Point() );
            }
        }
        tmp.push_back(_final_result);
    }
    else if(ui->radioButton_initialBoundary->isChecked())
    {
        tmp.push_back(_initial_boundary_img);
    }
    ui->qSmartGraphiicsView->initialize(tmp.size(), tmp[0].cols, tmp[0].rows);
    ui->qSmartGraphiicsView->setImage(tmp);
    ui->qSmartGraphiicsView->update();
}

void MainWindow::GVFFinished()
{
    if(_gvf_watcher.result())
    {
        ui->radioButton_resultImg->setEnabled(true);
        ui->radioButton_resultImg->setChecked(true);
        ui->radioButton_initialBoundary->setEnabled(true);
        ui->radioButton_initialBoundary->setChecked(true);
    }
    progressBar->setValue(100);
    progressBar->update();
    count.stop();
    gvfProgress = 0;
}

void MainWindow::updateProgressBar()
{
    progressBar->setValue(gvfProgress);
    progressBar->update();
}

void MainWindow::on_actionOpen_image_triggered()
{
    QString name = QFileDialog::getOpenFileName(this, "","");
    _opened_img = cv::imread(name.toStdString());
    if(_opened_img.empty())
    {
        QMessageBox::warning(this, "Error", "Fail to open image.");
        return;
    }

    _groundTruth.release();
    if(_gvf_watcher.isRunning())
        BMSD.stopWindow();

    if(ui->radioButton_originImg->isChecked())
    {
        ui->qSmartGraphiicsView->initialize(1, _opened_img.cols, _opened_img.rows);
        ui->qSmartGraphiicsView->setImage(_opened_img);
        ui->qSmartGraphiicsView->update();
        return;
    }
    ui->radioButton_originImg->setChecked(true);
}

void MainWindow::on_pushButton_process_clicked()
{
    if(_opened_img.empty())
        return;

    BMSD.setOTSUBeta(ui->doubleSpinBox_OTSUMan->value());
    BMSD.setBreastRatio(ui->doubleSpinBox_breastRatio->value());
    BMSD.setCLACHEClipLimit(ui->doubleSpinBox_clipLimit->value());
    BMSD.setCLACHEGridSize(cv::Size(ui->spinBox_gridSize->value(), ui->spinBox_gridSize->value()));
    BMSD.setFuzzyClusterNum(2);
    BMSD.setFuzzyFuzziness(ui->doubleSpinBox_fuzziness->value());
    BMSD.setFuzzyEpsilon(ui->doubleSpinBox_fuzzyEpsilon->value());
    BMSD.setFuzzyIteration(ui->spinBox_fuzzyIter->value());
    BMSD.setFuzzyDistType(static_cast<SoftCDistType>(ui->comboBox_softCDistType->currentIndex()));
    BMSD.setFuzzyInitType(static_cast<SoftCInitType>(ui->comboBox_softCInitType->currentIndex()));
    BMSD.setSnakeParams(ui->doubleSpinBox_snakeAlpha->value(),
                        ui->doubleSpinBox_snakeBeta->value(),
                        ui->doubleSpinBox_snakeGamma->value(),
                        ui->doubleSpinBox_snakeKappa->value());

    ui->radioButton_resultImg->setEnabled(false);
    ui->radioButton_initialBoundary->setEnabled(false);
    ui->radioButton_originImg->setChecked(true);
    if(_gvf_watcher.isRunning())
        BMSD.stopWindow();
    ui->statusBar->showMessage(QString("Set image."));
    BMSD.setImage(_opened_img);
    ui->statusBar->showMessage(QString("HAAR transform."));
    BMSD.HAARTransform();
    ui->statusBar->showMessage(QString("Breast ROI calculating."));
    BMSD.ROI();
    ui->statusBar->showMessage(QString("CLACHE."));
    BMSD.CLACHE();
    ui->statusBar->showMessage(QString("Fuzzy clustering."));
    BMSD.fuzzyClustering();
    ui->statusBar->showMessage(QString("Finding contour."));
    count.start();
    _gvf_future = QtConcurrent::run(&BMSD, &BreastMassSpiculatioDetect::acTiveContourModel);
    _gvf_watcher.setFuture(_gvf_future);
}

void MainWindow::on_actionRead_Groundtruth_triggered()
{
    QString name = QFileDialog::getOpenFileName(this, "","");
    _groundTruth = cv::imread(name.toStdString());
    if(_groundTruth.empty())
    {
        QMessageBox::warning(this, "Error", "Fail to open image.");
        return;
    }

    cv::cvtColor(_groundTruth, _groundTruth, cv::COLOR_BGR2GRAY);
}
