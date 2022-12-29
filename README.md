# FSIV-2122
Directorio de la asignatura de Fundamentos de Sistemas en Vision

# Funciones de test: Realzar el primer plano

/**
 * @brief Convierte una imagen en niveles de gris a color RGB.
 * @param img es la imagen en niveles de gris.
 * @return la imagen de entrada en el espacio RGB.
 */

cv::Mat convert_gray_to_rgb(const cv::Mat& img)
{
     
    cv::Mat out;
    cv::cvtColor(img,out,cv::COLOR_GRAY2BGR);  
    return out;
  
}

/**
 * @brief Convierte una imagen color BGR a niveles de gris.
 * @param img es la imagen en color BGR.
 * @return la imagen de entrada en niveles de gris.
 */
 
 cv::Mat convert_rgb_to_gray(const cv::Mat& img)
{

    cv::Mat out;    
    cv::cvtColor(img,out,cv::COLOR_BGR2GRAY);    
    return out;
}

/**
 * @brief Genera una máscara 0/255 con una ROI rectangular.
 * @param img_widht ancho de la imagen.
 * @param img_height alto de la imagen.
 * @param x coordenada x de la esquina superior izq. de la ROI
 * @param y coordenada y de la esquina superior izq. de la ROI
 * @param rect_widht ancho de la ROI.
 * @param rect_height alto de la ROI.
 * @param type es el tipo de Mat a crear.
 * @return la máscara generada.
 */
 
 cv::Mat generate_rectagle_mask(int img_width, int img_height, int x, int y, int rect_width, int rect_height, int type)
{

     cv::Mat mask = cv::Mat::zeros(img_height,img_width, type);;
     cv::rectangle(mask, cv::Point(x,y), cv::Point(x+rect_width,y+rect_height), cv::Scalar(255,255,255), cv::FILLED);
     return mask;
}

/**
 * @brief Genera una máscara 0/255 con una ROI circular.
 * @param img_widht ancho de la imagen.
 * @param img_height alto de la imagen.
 * @param x coordenada x del centro de la ROI
 * @param y coordenada y del centro de la ROI
 * @param radius de la ROI
 * @param type es el tipo de Mat a crear.
 * @return la máscara.
 */
 
 cv::Mat generate_circle_mask(int img_width, int img_height, int x, int y, int radius, int type)
{
    
    cv::Mat mask = cv::Mat::zeros(img_height, img_width, type);
    cv::circle(mask, cv::Point(x,y), radius, cv::Scalar(255,255,255), cv::FILLED);
    return mask;
}

/**
 * @brief Genera una máscara 0/255 con una ROI poligonal.
 * @param img_widht ancho de la imagen.
 * @param img_height alto de la imagen.
 * @param points es un vector con los vértices del polígono.
 * @param type es el tipo de Mat a crear.
 * @return la máscara generada.
 */
 
cv::Mat generate_polygon_mask(int img_width, int img_height, std::vector<cv::Point>& points, int type)
{
    
    std::vector< std::vector<cv::Point> > polys;
    cv::Mat mask = cv::Mat::zeros(img_height, img_width, type);
    polys.push_back(points);
    cv::fillPoly(mask, polys, cv::Scalar(255,255,255));
    return mask;
}

/**
 * @brief Realiza una combinación "hard" entre dos imágenes usando una máscara.
 * @param foreground la imagen "primer plano".
 * @param background la imagen "fondo".
 * @param mask la máscara 0 (fondo) / 255 (primer plano).
 * @return la imagen resultante de la combinación.
 */
 
cv::Mat combine_images(const cv::Mat& foreground, const cv::Mat& background,const cv::Mat& mask)
{
    
    cv::Mat output;
    cv::bitwise_and(foreground, mask,foreground);
    cv::bitwise_not(mask, mask);
    cv::bitwise_and(background, mask, background);
    cv::bitwise_or(foreground, background,output);
    return output;
}

# Funciones test: cbg_process

/**
 * @brief Convierte una imagen con tipo byte a flotante [0,1].
 * @param img imagen de entrada.
 * @param out imagen de salida.
 * @return la imagen de salida.
 * @warning la imagen de entrada puede ser monocroma o RGB.
 */
 
cv::Mat convert_image_byte_to_float(const cv::Mat& img, cv::Mat& out)
{
 
     img.convertTo(out,CV_32F, 1.0/255.0);
     return out;
}

/**
 * @brief Convierte una imagen con tipo float [0,1] a byte [0,255].
 * @param img imagen de entrada.
 * @param out imagen de salida.
 * @return la imagen de salida.
 * @warning la imagen de entrada puede ser monocroma o RGB.
 */
 
cv::Mat convert_image_float_to_byte(const cv::Mat& img, cv::Mat& out)
{
    
    img.convertTo(out, CV_8U, 255.0);    
    return out;
}

/**
 * @brief Convierte una imagen en color BGR a HSV.
 * @param img imagen de entrada.
 * @param out imagen de salida.
 * @return la imagen de salida.
 */
 
cv::Mat convert_bgr_to_hsv(const cv::Mat& img, cv::Mat& out)
{
    
    cv::cvtColor(img, out, cv::COLOR_BGR2HSV);
    return out;
}

/**
 * @brief Convierte una imagen en color HSV a BGR.
 * @param img imagen de entrada.
 * @param out imagen de salida.
 * @return la imagen de salida.
 */
 
cv::Mat convert_hsv_to_bgr(const cv::Mat& img, cv::Mat& out)
{
    
    cv::cvtColor(img,out,cv::COLOR_HSV2BGR);
    return out;
}

/**
 * @brief Realiza un control del brillo/contraste/gamma de la imagen.
 *
 * El proceso sería: O = c * I^g + b
 * 
 * Si la imagen es RGB y el flag only_luma es true, se utiliza el espacio HSV
 * para procesar sólo el canal V (luma).
 *
 * @param img  imagen de entrada.
 * @param out  imagen de salida.
 * @param contrast controla el ajuste del contraste.
 * @param brightness controla el ajuste del brillo.
 * @param gamma controla el ajuste de la gamma.
 * @param only_luma si es true sólo se procesa el canal Luma.
 * @return la imagen procesada.
 */

cv::Mat cbg_process (const cv::Mat & in, cv::Mat& out, double contrast, double brightness, double gamma, bool only_luma)
{
    
    std::vector<cv::Mat> canales;
    convert_image_byte_to_float(in,out);
    if(only_luma && in.channels()==3) convert_bgr_to_hsv(out,out);
    cv::split(out,canales);
    if(only_luma && in.channels() == 3){
        cv::pow(canales[2], gamma, canales[2]);
        canales[2] = contrast * canales[2] + brightness;
    }
    else{
        for(size_t i=0; i<canales.size(); i++){
            cv::pow(canales[i], gamma, canales[i]);
            canales[i] = contrast * canales[i] + brightness;
        }
    }
    cv::merge(canales, out);
    if(only_luma && in.channels() == 3) convert_hsv_to_bgr(out,out);
    convert_image_float_to_byte(out,out);
    return out;
}
