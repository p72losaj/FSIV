#include "common_code.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

cv::Mat fsiv_color_rescaling(const cv::Mat& in, const cv::Scalar& from, const cv::Scalar& to)
{
    CV_Assert(in.type()==CV_8UC3);
    cv::Mat out;
    //TODO
    //Cuidado con dividir por cero.
    //Evita los bucles.

    cv::Scalar rescaling;
    cv::divide(to, from, rescaling);
    out=in.mul(rescaling);

    //
    CV_Assert(out.type()==in.type());
    CV_Assert(out.rows==in.rows && out.cols==in.cols);
    return out;
}

cv::Mat fsiv_wp_color_balance(cv::Mat const& in)
{
    CV_Assert(in.type()==CV_8UC3);
    cv::Mat out;
    //TODO
    //Sugerencia: utiliza el espacio de color GRAY para
    //saber la ilumimancia de un pixel.

    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    double min, max;
    cv::Point minp, maxp;
    cv::minMaxLoc(out , &min, &max, &minp, &maxp);
    cv::Scalar white = in.at<cv::Vec3b>(maxp);
    out = fsiv_color_rescaling(in, white, cv::Scalar(255,255,255));
    //
    CV_Assert(out.type()==in.type());
    CV_Assert(out.rows==in.rows && out.cols==in.cols);
    return out;
}

cv::Mat fsiv_gw_color_balance(cv::Mat const& in)
{
    CV_Assert(in.type()==CV_8UC3);
    cv::Mat out;
    //TODO

    cv::Scalar grey = cv::mean(in);
    out = fsiv_color_rescaling(in, grey, cv::Scalar(128,128,128));

    //
    CV_Assert(out.type()==in.type());
    CV_Assert(out.rows==in.rows && out.cols==in.cols);
    return out;
}

cv::Mat fsiv_color_balance(cv::Mat const& in, float p)
{
    CV_Assert(in.type()==CV_8UC3);
    CV_Assert(0.0f<p && p<100.0f);
    cv::Mat out;
    //TODO
    //Sugerencia: utiliza el espacio de color GRAY para
    //saber la ilumimancia de un pixel.

    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    cv::Mat hist;
    int histSize[]= {256}, canales[] = {0}, i=0;
    float intensityRanges[] = {0, 256}, accum=0.0, percent= 1.0 - (p/100.0);
    const float* ranges[] = {intensityRanges};
    cv::calcHist(&out, 1, canales, cv::Mat(), hist, 1, histSize, ranges, true, false);
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1, -1, cv::Mat());
    while(i < hist.rows && percent >= accum){
        accum+=hist.at<float>(i, 0);
        i++;
    }
    cv::Scalar mean = cv::mean(in, out >= i);
    out = fsiv_color_rescaling(in, mean, cv::Scalar(255, 255, 255));
    //
    CV_Assert(out.type()==in.type());
    CV_Assert(out.rows==in.rows && out.cols==in.cols);
    return out;
}
