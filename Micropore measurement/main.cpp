#include "Microporemeasurement.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Microporemeasurement w;
    w.show();
    return a.exec();
}
