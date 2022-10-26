#include <iostream>
#include "common_code.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat
fsiv_create_gaussian_filter(const int r)
{
    CV_Assert(r>0);
    cv::Mat ret_v;

    //TODO: Remenber 6*sigma is approx 99,73% of the distribution.
    ret_v = cv::Mat::zeros((2*r+1), (2*r+1), CV_32FC1);
    float sigma = (2*r+1)/6.0;
    float exp;
    for(int i = -r; i <= r; i++){
        for(int j = -r; j <= r; j++){
            cv::Point point(r+i, r+j);
            exp = (pow(i,2) + pow(j,2));
            ret_v.at<float>(point) = std::exp(-(exp/(2*(pow(sigma, 2)))));
        }
    }

    float accum = static_cast<float>(cv::sum(ret_v)[0]);
    if(accum != 0)
        ret_v/= accum;


    //
    CV_Assert(ret_v.type()==CV_32FC1);
    CV_Assert(ret_v.rows==(2*r+1) && ret_v.rows==ret_v.cols);
    CV_Assert(std::abs(1.0-cv::sum(ret_v)[0])<1.0e-6);
    return ret_v;
}

cv::Mat
fsiv_extend_image(const cv::Mat& img, const cv::Size& new_size, int ext_type)
{
    CV_Assert(img.rows<new_size.height);
    CV_Assert(img.cols<new_size.width);
    cv::Mat out;
    //TODO
    //Hint: use cv::copyMakeBorder()

    // new_size.height = img.rows + 2*r ->r = new_size.height-img.rows)/2
    int r = (new_size.height-img.rows)/2;

    if(ext_type == 0){
         cv::copyMakeBorder(img,out,r,r,r,r,cv::BORDER_CONSTANT);
    }

    else{
        cv::copyMakeBorder(img,out,r,r,r,r,cv::BORDER_WRAP);
    }

    //
    CV_Assert(out.type()==img.type());
    CV_Assert(out.rows == new_size.height);
    CV_Assert(out.cols == new_size.width);
    return out;
}

cv::Mat
fsiv_create_sharpening_filter(const int filter_type, int r1, int r2)
{
    CV_Assert(0<=filter_type && filter_type<=2);
    CV_Assert(0<r1 && r1<r2);
    cv::Mat filter;
    //TODO
    //Remenber DoG = G[r2]-G[r1].
    //Hint: use fsiv_extend_image() to extent G[r1].
    filter = cv::Mat::zeros(3, 3, CV_32FC1);
    if(filter_type==0){
          filter.at<float>(0,1) = -1;
          filter.at<float>(1,0) = -1;
          filter.at<float>(1,1) = 5;
          filter.at<float>(1,2) = -1;
          filter.at<float>(2,1) = -1;
      }


    // Filtro laplaciano en lap8

    else if(filter_type==1){
        filter.at<float>(0,0) = -1;
        filter.at<float>(0,1) = -1;
        filter.at<float>(0,2) = -1;
        filter.at<float>(1,0) = -1;
        filter.at<float>(1,1) = 9;
        filter.at<float>(1,2) = -1;
        filter.at<float>(2,0) = -1;
        filter.at<float>(2,1) = -1;
        filter.at<float>(2,2) = -1;
      }
    // Filtro laplaciano DOG

    else{
            cv::Mat g1 = fsiv_create_gaussian_filter(r1);
            cv::Mat g2 = fsiv_create_gaussian_filter(r2);
            g1 = fsiv_extend_image(g1, g2.size(),0);
            filter = g1-g2;
            filter.at<float>(g2.rows/2, g2.cols/2) += 1;
        }


    //
    CV_Assert(filter.type()==CV_32FC1);
    CV_Assert((filter_type == 2) || (filter.rows==3 && filter.cols==3) );
    CV_Assert((filter_type != 2) || (filter.rows==(2*r2+1) &&
                                     filter.cols==(2*r2+1)));
    return filter;
}



cv::Mat
fsiv_image_sharpening(const cv::Mat& in, int filter_type, bool only_luma,
                      int r1, int r2, bool circular)
{
    CV_Assert(in.depth()==CV_8U);
    CV_Assert(0<r1 && r1<r2);
    CV_Assert(0<=filter_type && filter_type<=2);
    cv::Mat out;
    //TODO
    //Hint: use cv::filter2D.
    //Remenber: if circular, first the input image must be circular extended,
    //  and then clip the result.

    // Creamos el filtro

    cv::Mat filter = fsiv_create_sharpening_filter(filter_type, r1, r2);
    cv::Size new_size (in.cols+2*r2, in.rows+2*r2);

    if(!only_luma){
      cv::Mat img;
      if(circular){
          img = fsiv_extend_image(in,new_size,1);
          cv::filter2D(img, out, -1, filter);
          out = out(cv::Rect(r2,r2,in.cols,in.rows));
      }

      else{
          img = fsiv_extend_image(in,new_size,0);
          cv::filter2D(in, img, -1, filter);
          out = img.clone();
      }
    }
    // Canal luma
    else{
        cv::Mat img = in.clone();
        cv::cvtColor(img,img,cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> canales;
        cv::split(img,canales);

        if(circular){
            canales[2] = fsiv_extend_image(canales[2], new_size,1);
            cv::filter2D(canales[2], canales[2], -1, filter);
        }
        else{
            //canales[2] = fsiv_extend_image(canales[2], new_size, 0);
        }

        cv::merge(canales,out);
        cv::cvtColor(out,out,cv::COLOR_HSV2BGR);
    }
    //
    CV_Assert(out.type()==in.type());
    CV_Assert(out.size()==in.size());
    return out;
}
