#pragma once
//º¯ÊıÃûÉùÃ÷
#include <QtWidgets/QMainWindow>
#include "ui_Microporemeasurement.h"
class Microporemeasurement : public QMainWindow
{
    Q_OBJECT

public:
    Microporemeasurement(QWidget *parent = nullptr);
    void ClickedToCompute();
    ~Microporemeasurement();

private:
    Ui::MicroporemeasurementClass ui;
};
