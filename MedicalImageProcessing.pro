#-------------------------------------------------
#
# Project created by QtCreator 2015-12-12T15:30:48
#
#-------------------------------------------------

QT       += core gui svg

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MedicalImageProcessing
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    qsmartgraphicsview.cpp \
    breastmassspiculatiodetect.cpp \
    fuzzy_clustering.cpp \
    activeContour/common.cpp \
    activeContour/gvfc.cpp

HEADERS  += mainwindow.h \
    qsmartgraphicsview.h \
    breastmassspiculatiodetect.h \
    fuzzy_clustering.hpp \
    activeContour/common.h \
    activeContour/gvfc.h

FORMS    += mainwindow.ui

RESOURCES += \
    octicons/octicons.qrc

win32::LIBS += -lpsapi

msvc {
  QMAKE_CXXFLAGS += -openmp -arch:AVX -D "_CRT_SECURE_NO_WARNINGS"
  QMAKE_CXXFLAGS_RELEASE *= -O2
}

INCLUDEPATH += $$quote(D:/libraries/opencv300/include)\
               $$PWD\

OPENCVLIB += $$quote(D:/libraries/opencv300/x64/vc12/lib)


CONFIG(debug, debug|release){
LIBS+= $$OPENCVLIB/opencv_world300d.lib\
       $$OPENCVLIB/opencv_ts300d.lib\
       #$$OPENCVLIB/opencv_saliency300.lib

}

CONFIG(release, debug|release){
LIBS+= $$OPENCVLIB/opencv_world300.lib\
       $$OPENCVLIB/opencv_ts300.lib\
       #$$OPENCVLIB/opencv_saliency300.lib
}

DEFINES += HAVE_OPENCV NO_SIDEMENU
