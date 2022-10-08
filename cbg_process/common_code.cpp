#include "common_code.hpp"

cv::Mat
convert_image_byte_to_float(const cv::Mat& img, cv::Mat& out)
{
    CV_Assert(img.depth()==CV_8U);
    //TODO
        img.convertTo(out,CV_32F, 1.0/255.0);
    //
    CV_Assert(out.rows==img.rows && out.cols==img.cols);
    CV_Assert(out.depth()==CV_32F);
    CV_Assert(img.channels()==out.channels());
    return out;
}

cv::Mat
convert_image_float_to_byte(const cv::Mat& img, cv::Mat& out)
{
    CV_Assert(img.depth()==CV_32F);
    //TODO
        img.convertTo(out,CV_8U,255.0);
    //
    CV_Assert(out.rows==img.rows && out.cols==img.cols);
    CV_Assert(out.depth()==CV_8U);
    CV_Assert(img.channels()==out.channels());
    return out;
}

cv::Mat
convert_bgr_to_hsv(const cv::Mat& img, cv::Mat& out)
{
    CV_Assert(img.channels()==3);
    //TODO
        cv::cvtColor(img,out, cv::COLOR_BGR2HSV);
    //
    CV_Assert(out.channels()==3);
    return out;
}

cv::Mat
convert_hsv_to_bgr(const cv::Mat& img, cv::Mat& out)
{
    CV_Assert(img.channels()==3);
    //TODO
        cv::cvtColor(img,out,cv::COLOR_HSV2BGR);
    //
    CV_Assert(out.channels()==3);
    return out;
}

cv::Mat
cbg_process (const cv::Mat & in, cv::Mat& out,
             double contrast, double brightness, double gamma,
             bool only_luma)
{
    CV_Assert(in.depth()==CV_8U);
    //TODO
    //Recuerda: es recomendable trabajar en flotante [0,1]
    //Después deshacer el cambio a byte [0,255]

cv::Mat inf;
    std::vector<cv::Mat> canales; // Vector de canales
    convert_image_byte_to_float(in, inf); // Convertimos imagen in en float
    // Caso 1: Control luna

    if(only_luma && in.channels() == 3){
        cv::Mat hsv = inf.clone(); // Imagen hsv
        convert_bgr_to_hsv(inf,hsv); // Convertimos la imagen hsv en formato hsv
        cv::split(hsv,canales); // Dividimos la imagen hsv en canales
        cv::pow(canales[2], gamma, canales[2]);
        canales[2] = (contrast * canales[2]) + brightness;
        cv::merge(canales,hsv); // Unimos los canales de la imagen hsv
        convert_hsv_to_bgr(hsv,inf);// Convertimos la imagen hsv en formato bgr
    }

    // sin control luma

    else{
        cv::split(inf,canales); // Dividimos la imagen en canales
        for(size_t i=0; i < canales.size(); i++){
            cv::pow(canales[i], gamma, canales[i]);
            canales[i] = (contrast * canales[i]) + brightness;
        }
        cv::merge(canales,inf);// Unimos los canales de la imagen
    }

    convert_image_float_to_byte(inf,out);// Convertimos la imagen de valores flotantes en valores enteros

    //
    CV_Assert(out.rows==in.rows && out.cols==in.cols);
    CV_Assert(out.depth()==CV_8U);
    CV_Assert(out.channels()==in.channels());
    return out;
}
