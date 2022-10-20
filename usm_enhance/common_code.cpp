#include "common_code.hpp"

cv::Mat
fsiv_create_box_filter(const int r)
{
    CV_Assert(r>0);
    cv::Mat ret_v;
    //TODO
    float size = 2*r+1;
    ret_v = cv::Mat::ones(size, size, CV_32F) / pow(size,2);
    //
    CV_Assert(ret_v.type()==CV_32FC1);
    CV_Assert(ret_v.rows==(2*r+1) && ret_v.rows==ret_v.cols);
    CV_Assert(std::abs(1.0-cv::sum(ret_v)[0])<1.0e-6);
    return ret_v;
}

cv::Mat
fsiv_create_gaussian_filter(const int r)
{
    CV_Assert(r>0);
    cv::Mat ret_v;
    //TODO: Remenber 6*sigma is approx 99,73% of the distribution.

    float size = 2*r+1;
    float a = 1.0/(2.0*M_PI*pow(size/6.0,2));
    ret_v = cv::Mat(size, size, CV_32FC1);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float b = -(pow(i-r,2)+pow(j-r,2))/(2.0*pow(size/6.0,2));
            ret_v.at<float>(i, j) = a * exp(b);
            }
        }
    ret_v /= (cv::sum(ret_v));

    //
    CV_Assert(ret_v.type()==CV_32FC1);
    CV_Assert(ret_v.rows==(2*r+1) && ret_v.rows==ret_v.cols);
    CV_Assert(std::abs(1.0-cv::sum(ret_v)[0])<1.0e-6);
    return ret_v;
}

cv::Mat
fsiv_fill_expansion(cv::Mat const& in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r>0);
    cv::Mat ret_v;
    //TODO:
    //Use of cv::copyMakeBorder is not allowed.
    //Hint you don't need use any for sentence.

    if(in.type() == CV_32FC1){
        ret_v = cv::Mat::zeros(in.rows+2*r, in.cols+2*r, CV_32FC1);
    }

    else{ 
        ret_v = cv::Mat::zeros(in.rows+2*r, in.cols+2*r, CV_8UC1);
    }

    cv::Rect rectangulo(r, r, in.cols, in.rows); // (x, y, width, height)
    in.copyTo(ret_v(rectangulo)); // copia la imagen en el centro de la imagen de salida
    //
    CV_Assert(ret_v.type()==in.type());
    CV_Assert(ret_v.rows == in.rows+2*r);
    CV_Assert(ret_v.cols == in.cols+2*r);
    return ret_v;
}

cv::Mat
fsiv_circular_expansion(cv::Mat const& in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r>0);
    cv::Mat ret_v;
    //TODO
    //Use of cv::copyMakeBorder is not allowed.
    //Hint you don't need use any "for" sentence, only 9 copyTo from "in"
    // rois to "ret_v" rois.

    // Expansion de la imagen original
    ret_v = fsiv_fill_expansion(in, r);    
    // expansion superior e inferior
    in(cv::Rect(0,0,in.cols,r)).copyTo(ret_v(cv::Rect(r, ret_v.rows-r, in.cols,r)));
    in(cv::Rect(0,in.rows-r,in.cols,r)).copyTo(ret_v(cv::Rect(r, 0, in.cols,r)));
    // expansion lateral
    in(cv::Rect(0,0,r,in.rows)).copyTo(ret_v(cv::Rect(ret_v.cols-r,r,r,in.rows)));
    in(cv::Rect(in.cols-r,0,r,in.rows)).copyTo(ret_v(cv::Rect(0,r,r,in.rows)));
    // expansion en las esquinas
    in(cv::Rect(0,0,r,r)).copyTo(ret_v(cv::Rect(ret_v.cols-r,ret_v.rows-r,r,r)));
    in(cv::Rect(in.cols-r,in.rows-r,r,r)).copyTo(ret_v(cv::Rect(0,0,r,r)));
    in(cv::Rect(in.cols-r,0,r,r)).copyTo(ret_v(cv::Rect(0,ret_v.rows-r,r,r)));
    in(cv::Rect(0,in.rows-r,r,r)).copyTo(ret_v(cv::Rect(ret_v.cols-r,0,r,r)));

    //
    CV_Assert(ret_v.type()==in.type());
    CV_Assert(ret_v.rows == in.rows+2*r);
    CV_Assert(ret_v.cols == in.cols+2*r);
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(0,0)==in.at<uchar>(in.rows-r, in.cols-r));
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(0,ret_v.cols/2)==in.at<uchar>(in.rows-r, in.cols/2));
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(0,ret_v.cols-1)==in.at<uchar>(in.rows-r, r-1));
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(ret_v.rows/2,0)==in.at<uchar>(in.rows/2, in.cols-r));
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(ret_v.rows/2,ret_v.cols/2)==in.at<uchar>(in.rows/2, in.cols/2));
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(ret_v.rows-1,0)==in.at<uchar>(r-1, in.cols-r));
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(ret_v.rows-1,ret_v.cols/2)==in.at<uchar>(r-1, in.cols/2));
    CV_Assert(!(in.type()==CV_8UC1) || ret_v.at<uchar>(ret_v.rows-1,ret_v.cols-1)==in.at<uchar>(r-1, r-1));
    return ret_v;
}

cv::Mat
fsiv_filter2D(cv::Mat const& in, cv::Mat const& filter)
{
    CV_Assert(!in.empty() && !filter.empty());
    CV_Assert(in.type()==CV_32FC1 && filter.type()==CV_32FC1);
    cv::Mat ret_v;
    //TODO
ret_v = cv::Mat::zeros((in.rows-2*(filter.rows/2)), (in.cols-2*(filter.cols/2)), CV_32FC1);
    // Aplicamos el fliltro 2D a la nueva imagen
    for (int i = 0; i < ret_v.rows; ++i) {
        for (int j = 0; j < ret_v.cols; ++j){
            ret_v.at<float>(i,j) = sum(filter.mul(in(cv::Rect(j, i, filter.cols, filter.rows)))).val[0];
        }
    }
    //
    CV_Assert(ret_v.type()==CV_32FC1);
    CV_Assert(ret_v.rows==in.rows-2*(filter.rows/2));
    CV_Assert(ret_v.cols==in.cols-2*(filter.cols/2));
    return ret_v;
}

cv::Mat
fsiv_combine_images(const cv::Mat src1, const cv::Mat src2,
                    double a, double b)
{
    CV_Assert(src1.type()==src2.type());
    CV_Assert(src1.rows==src2.rows);
    CV_Assert(src1.cols==src2.cols);
    cv::Mat ret_v;
    //TODO
  ret_v = src1.mul(a) + src2.mul(b);

    //
    CV_Assert(ret_v.type()==src2.type());
    CV_Assert(ret_v.rows==src2.rows);
    CV_Assert(ret_v.cols==src2.cols);
    return ret_v;
}

cv::Mat
fsiv_usm_enhance(cv::Mat  const& in, double g, int r,
                 int filter_type, bool circular, cv::Mat *unsharp_mask)
{
    CV_Assert(!in.empty());
    CV_Assert(in.type()==CV_32FC1);
    CV_Assert(r>0);
    CV_Assert(filter_type>=0 && filter_type<=1);
    CV_Assert(g>=0.0);
    cv::Mat ret_v;
    //TODO
    //Hint: use your own functions fsiv_xxxx

cv::Mat filtro, expansion;
    // Aplicamos los filtros
    if(filter_type == 0){
        filtro = fsiv_create_box_filter(r);
    }
    else{ 
        filtro = fsiv_create_gaussian_filter(r);
    }
    // Expansion de la imagen
    if(!circular){
        expansion = fsiv_fill_expansion(in, r);
    }
    else{
        expansion = fsiv_circular_expansion(in, r);
    }
    // Aplicamos el filtro 2D
    cv::Mat filtro2D = fsiv_filter2D(expansion, filtro);

    // unsharp  mask
    if(unsharp_mask){
        *unsharp_mask = filtro2D;
    }
    // Aplicamos la combinacion de imagenes
    ret_v = fsiv_combine_images(in, filtro2D, 1+g, -g);

    //
    CV_Assert(ret_v.rows==in.rows);
    CV_Assert(ret_v.cols==in.cols);
    CV_Assert(ret_v.type()==CV_32FC1);
    return ret_v;
}
