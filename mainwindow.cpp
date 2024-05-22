#include <iostream>

#include <QPixmap>
#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include "include/common.hpp"
#include "include/face_recognizer.hpp"


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QPixmap pix("/home/jetson/ssd/watchout/srcs/DMS.png");
    //int w = ui->label_pic->width();
    //int h = ui->label_pic->height();
    ui->label_pic->setPixmap(pix.scaled(500, 500, Qt::KeepAspectRatio));
    ui->status->setAlignment(Qt::AlignCenter);
    ui->status->setText("Current Status");
    ui->registButton->setText("Register");
    ui->authenticButton->setText("Authenticate");
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_registButton_clicked()
{
    dms::DriverRegistrar driver_registrar;
    bool flag_regist;
    int err;

    ui->status->setText("regist button click");
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
       std::cerr << "Unable to connect to camera" << std::endl;
       ui->status->setText("Unable to connect to camera");
    }
    std::vector<cv::Mat> main_cam_images;
    QString command[5] = {"카메라를 쳐다보세요", "30도 왼쪽을 보세요", "30도 오른쪽을 보세요", "30도 위를 보세요", "30도 아래를 보세요"};    
    std::vector<cv::Point2d> driver_gaze_angle;

    std::string driver_num = "1"; // 이거 필요없는거임. 저장은 1번부터 차례대로 해버릴꺼임. 그냥 함수에 매개변수 맞출려고 구색상 만든거
    for (int i = 0; i < 5; i++){
        ui->status->setText(command[i]); // 이거 setText에 되는지 모르겠음
        usleep(2000000);
        cv::Mat img_capture;
        cap >> img_capture; // 찰칵
        main_cam_images.push_back(img_capture);
    }
    flag_regist = driver_registrar.registerDriver(main_cam_images,driver_gaze_angle,driver_num,err);  //driver_num string으로 변환해야함
    if (flag_regist) {
       std::cout << "등록 성공" << std::endl;
       ui->status->setText("Successfully registered!");
    }
    else {
       std::cout << "등록 실패" << std::endl;
       ui->status->setText("Registration failed.");
    }
}


void MainWindow::on_authenticButton_clicked()
{
    dms::DriverAuthenticator driver_authenticator;

    ui->status->setText("authentic button click");

    dms::DriverRegistrar driver_registrar;
    std::string driver_name;
    int err;
    bool flag_authentic = false;

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
    std::cerr << "Unable to connect to camera" << std::endl;
    ui->status->setText("Unable to connect to camera");

    }

    cv::Mat img_capture;
    ui->status->setText("take photo");
    usleep(3000000);
    cap >> img_capture;
    
    flag_authentic = driver_authenticator.authenticateDriver(img_capture, driver_name, err);
    if(flag_authentic){
       ui->status->setText("Authenticated! Access granted.");
       QCoreApplication::quit(); //QT 종료
    }
    else{
       ui->status->setText("Authentication failed. Please try again.");
    }

    cap.release();
}
