#include "common_code.hpp"

cv::Mat
convert_gray_to_rgb(const cv::Mat& img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels()==1);
    cv::Mat out;

    //TODO
        cvtColor(img,out,cv::COLOR_GRAY2RGB);
    //

    CV_Assert(out.channels()==3);
    CV_Assert(img.rows==out.rows && img.cols==out.cols);
    CV_Assert(img.depth()==out.depth());
    return out;
}

cv::Mat
convert_rgb_to_gray(const cv::Mat& img)
{
    CV_Assert(!img.empty());
    CV_Assert(img.channels()==3);
    cv::Mat out;
    //TODO
        cvtColor(img,out,cv::COLOR_BGR2GRAY);
    //
    CV_Assert(out.channels()==1);
    CV_Assert(img.rows==out.rows && img.cols==out.cols);
    CV_Assert(img.depth()==out.depth());
    return out;
}

cv::Mat
generate_rectagle_mask(int img_width, int img_height,
                       int x, int y, int rect_width, int rect_height, int type)
{
    CV_Assert(img_width>0 && img_height>0);
    CV_Assert(x>=0 && x<img_width);
    CV_Assert(y>=0 && y<img_height);
    CV_Assert(rect_width>0 && rect_height>0);
    cv::Mat mask;
    //TODO
        mask = cv::Mat::zeros(img_height,img_width,type); // Generamos una imagen con valor de pixel 0
        cv::Point vertices(x,y);// Obtenemos los vertices
        cv::Point lados(x+rect_width,y+rect_height);// Obtenemos los lados
        cv::rectangle(mask,vertices,lados,cv::Scalar(255,255,255), cv::FILLED);// Rellenamos el rectangulo con el color gris

    //
    CV_Assert(mask.rows==img_height && mask.cols==img_width);
    CV_Assert(mask.type()==type);
    return mask;
}

cv::Mat
generate_circle_mask(int img_width, int img_height,
                     int x, int y, int radius, int type)
{
    CV_Assert(img_width>0 && img_height>0);
    CV_Assert(x>=0 && x<img_width);
    CV_Assert(y>=0 && y<img_height);
    CV_Assert(radius > 0);
    cv::Mat mask;
    //TODO
    mask = cv::Mat::zeros(img_height,img_width,type); // Generamos una imagen con valor de pixel 0
    cv::Point centro(x,y);// Punto centro del circulo
    cv::circle(mask,centro,radius,cv::Scalar(255,255,255), cv::FILLED);// Generamos el circulo
    //
    CV_Assert(mask.rows==img_height && mask.cols==img_width);
    CV_Assert(mask.type()==type);
    return mask;

}

cv::Mat
generate_polygon_mask(int img_width, int img_height,
                      std::vector<cv::Point>& points, int type)
{
    CV_Assert(img_width>0 && img_height>0);
    CV_Assert(points.size()>=3);
    cv::Mat mask;
    std::vector< std::vector<cv::Point> > polys;
    //TODO

    mask = cv::Mat::zeros(img_height,img_width,type); // Generamos la mascara
    polys.push_back(points);
    cv::fillPoly(mask, polys, cv::Scalar(255,255,255),type);// Generamos el poligono
    //
    CV_Assert(mask.rows==img_height && mask.cols==img_width);
    CV_Assert(mask.type()==type);
    return mask;
}

cv::Mat
combine_images(const cv::Mat& foreground, const cv::Mat& background,
               const cv::Mat& mask)
{
    CV_Assert(background.rows == foreground.rows &&
              background.cols==foreground.cols);
    CV_Assert(background.type()==foreground.type());
    CV_Assert(mask.rows == foreground.rows &&
              mask.cols==foreground.cols);
    CV_Assert(mask.type()==foreground.type());
    cv::Mat output;
    //TODO

    cv::Mat mascaraUnion1,mascaraUnion2, mascaraNegada;

    cv::bitwise_and(foreground,mask,mascaraUnion1); // Unimos la mascara a la imagen foreground

    cv::bitwise_not(mask,mascaraNegada); // Negamos la mascara

    cv::bitwise_and(background,mascaraNegada,mascaraUnion2); // Unimos la imagen background con la mascara negada

    cv::bitwise_or(mascaraUnion1,mascaraUnion2,output);// Generamos la imagen de salida buscada

    //
    CV_Assert(output.rows == foreground.rows && output.cols==foreground.cols);
    CV_Assert(output.type()==foreground.type());
    return output;
}
