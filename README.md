# FSIV-2223

Funciones de test de las practicas

# Realzar el primer plano

/**
 * @brief Convierte una imagen en niveles de gris a color RGB.
 * @param img es la imagen en niveles de gris.
 * @return la imagen de entrada en el espacio RGB.
 */

cv::Mat convert_gray_to_rgb(const cv::Mat& img)
{
     
    // 1. Convertimos de gray a RGB
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

    // 1. Convertimos de rgb a gray
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
     
     // 1. Generamos una mascara vacia
     cv::Mat mask = cv::Mat::zeros(img_height,img_width, type);
     // 2. Rellenamos la mascara aplicando ROI rectangular
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
    
    // 1. Generamos una mascara vacia
    cv::Mat mask = cv::Mat::zeros(img_height, img_width, type);
    // 2. Generamos la mascara con una ROI circular
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
    
    // 1. Generamos una mascara vacia
    cv::Mat mask = cv::Mat::zeros(img_height, img_width, type);
    // 2. Obtenemos los puntos del poligono
    std::vector< std::vector<cv::Point> > polys;
    polys.push_back(points);
    // 3. Generamos la mascara con una ROI poligonal
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
    // 1. Obtenemos el primer plano de la imagen
    cv::bitwise_and(foreground, mask,foreground);
    // 2. Negamos la mascara
    cv::bitwise_not(mask, mask);
    // 3. Obtenemos el fondo de la imagen
    cv::bitwise_and(background, mask, background);
    // 4. Combinamos el primer plano y el fondo de la imagen
    cv::bitwise_or(foreground, background,output);
    return output;
}

# cbg_process

/**
 * @brief Convierte una imagen con tipo byte a flotante [0,1].
 * @param img imagen de entrada.
 * @param out imagen de salida.
 * @return la imagen de salida.
 * @warning la imagen de entrada puede ser monocroma o RGB.
 */
 
cv::Mat convert_image_byte_to_float(const cv::Mat& img, cv::Mat& out)
{
    
    // 1. Imagen de tipo byte a flotante
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
    
    // 1. Imagen de tipo float a byte
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
    
    // 1. Imagen BGR a HSV
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
    
    // 1. Imagen HSV a BGR
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
    
    // 1. Imagen de salida en valores flotante
    convert_image_byte_to_float(in,out);
    // Opcion luma -> imagen salida en hsv
    if(only_luma && in.channels()==3) convert_bgr_to_hsv(out,out); 
    // 2. Dividimos la imagen en canales
    std::vector<cv::Mat> canales;
    cv::split(out,canales);
    // 3. Opcion luma -> Procesamos el canal luma
    if(only_luma && in.channels() == 3){
        cv::pow(canales[2], gamma, canales[2]);
        canales[2] = contrast * canales[2] + brightness;
    }
    // 3. Procesamos todos los canales
    else{
        for(size_t i=0; i<canales.size(); i++){
            cv::pow(canales[i], gamma, canales[i]);
            canales[i] = contrast * canales[i] + brightness;
        }
    }
    // 4. Unimos los canales
    cv::merge(canales, out);
    // Opcion luma -> Convertimos la imagen en BGR
    if(only_luma && in.channels() == 3) convert_hsv_to_bgr(out,out);
    // 5. Convertimos la imagen a byte
    convert_image_float_to_byte(out,out);
    return out;
}

# img_equalization

/**
 * @brief Calcula el histograma de una imagen monocroma.
 * @param in es la imagen con formato 8UC1.
 * @param hist almacena el histograma calculado.
 * @return el histograma calculado.
 * @pre in.type()==CV_8UC1
 * @pre hist.empty()||(hist.type()==CV_32FC1 && hist.rows==256 && hist.cols==1)
 * @post hist.type()==CV_32FC1
 * @post hist.rows==256 && hist.cols==1
 */
 
cv::Mat fsiv_compute_histogram(const cv::Mat& in, cv::Mat& hist)
{
    
    // 1. Calculamos el histograma
    int histSize[]= {256}, channels[] = {0};
    float intensityRanges[] = {0, 256};
    const float* ranges[] = {intensityRanges};
    cv::calcHist(&in, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    return hist;
}

/**
 * @brief Normaliza el histogram de forma que su suma sea 1.0
 * @param hist el histograma a normalizar.
 * @pre hist.type()==CV_32FC1
 * @pre hist.rows==256 && hist.cols==1
 * @post hist.type()==CV_32FC1
 * @post hist.rows==256 && hist.cols==1
 * @post sum(hist)[0]==0.0 || abs(sum(hist)[0]-1.0)<=1.0e-6
 */
 
void fsiv_normalize_histogram(cv::Mat& hist)
{
    
    // 1. Normalizamos el histograma
    cv::normalize(hist,hist,1,0, cv::NORM_L1, -1, cv::Mat());
}

/**
 * @brief acumula el histograma.
 * @param hist el histograma a acumular.
 * @pre hist.type()==CV_32FC1
 * @pre hist.rows==256 && hist.cols==1
 * @post hist.type()==CV_32FC1
 * @post hist.rows==256 && hist.cols==1
 * @post sum(old.hist)==0.0 || abs(sum(old.hist)-hist[255])/sum(old.hist) <= 1.0e-5
 */

void fsiv_accumulate_histogram(cv::Mat& hist)
{
    
    // 1. Calculamos el histograma acumulado
    for(size_t i=0; i<hist.rows; i++) hist.at<float>(i) += hist.at<float>(i-1);
}

/**
 * @brief Crea una tabla para realizar la ecualización de una imagen.
 * @param hist es el histograma de la imagen.
 * @param hold_median si es true, la mediana no se transformará.
 * @return la tabla creada.
 * @pre hist.type()==CV_32FC1
 * @pre hist.rows==256 && hist.cols==1
 * @post retval.type()==CV_8UC1
 * @post retval.rows==256 && retval.cols==1 
 */
 
cv::Mat fsiv_create_equalization_lookup_table(const cv::Mat& hist, bool hold_median)
{
    
    cv::Mat lkt = hist.clone();
    // 1. Normalizamos el histograma
    fsiv_normalize_histogram(lkt);
    // 2. Calculamos el histograma acumulado
    fsiv_accumulate_histogram(lkt);
    // 3.1.Opcion hold median
    if(hold_median){ 
        // 4. Obtenemos la posicion media del histograma
        int median = 0;
        float position=0;
        while(position < 0.5 && median < 256){
            median++;
            position = lkt.at<float>(median);
        }
        // 5. Obtenemos las 2 mitades del histograma
        if(0<median<255){
            float medianf = median;
            // Obtenemos el rango de las 2 mitades del histograma
            cv::Range first_half(0,median), second_half(median,256);
            // Primera mitad del histograma
            lkt(first_half, cv::Range::all()) /= position ; 
            lkt(first_half, cv::Range::all()) *= median;
            // Segunda mitad del histograma
            lkt(second_half, cv::Range::all()) -= position;
            lkt(second_half, cv::Range::all()) /= (1-position) ;
            lkt(second_half, cv::Range::all())*=255.0-median;
            lkt(second_half, cv::Range::all())+=medianf;
        }
        // 6. Convertimos la imagen a CV_8UC1
        lkt.convertTo(lkt, CV_8UC1);
    }
    // 3.2.Opcion sin hold median -> Convertimos la imagen a CV_8UC1
    else lkt.convertTo(lkt, CV_8UC1, 255.0);
    return lkt;
}

/**
 * @brief Aplica una "lookup table"
 * @param in la imagen de entrada.
 * @param lkt la tabla.
 * @param out la imgen de salida.
 * @return la imagen de salida.
 * @pre in.type()==CV_8UC1
 * @pre lkt.type()==CV_8UC1
 * @pre lkt.rows==256 && lkt.cols==1
 * @pre out.empty() || (out.type()==CV_8UC1 && out.rows==in.rows && out.cols==in.cols)
 * @post out.rows ==in.rows && out.cols==in.cols && out.type()==in.type()
 */
 
cv::Mat fsiv_apply_lookup_table(const cv::Mat&in, const cv::Mat& lkt,cv::Mat& out)
{

    // 1. Aplicamos un lookup table
    cv::LUT(in,lkt,out);
    return out;
}

/**
 * @brief Ecualiza una imagen.
 * @param in es la imagen a ecualizar.
 * @param out es la imagen ecualizada.
 * @param hold_median si es cierto la mediana se transformá al mismo valor.
 * @param radius si es >0, se aplica ecualización local con ventanas de radio r.
 * @return la imagen ecualizada.
 * @warning si se aplica procesado local, el área de la imagen de entrada que
 * no puede ser procesada se copia directamente en la salida.
 */
 
cv::Mat fsiv_image_equalization(const cv::Mat& in, cv::Mat& out, bool hold_median, int radius)
{
    
    cv::Mat hist, lkt;
    out = in.clone();
    // 1.1. Aplicamos procesamiento lucal
    if(radius > 0){
        for(int i = 0; i <= in.rows - (2*radius+1); i++){
            for(int j = 0; j <= in.cols - (2*radius+1); j++){
                // 2. Obtenemos una ventana de radio r
                cv::Mat ventana =  in(cv::Rect(j, i, 2*radius+1, 2*radius+1));
                // 3. Aplicamos la computacion del histograma
                fsiv_compute_histogram(ventana, hist);
                // 4. Creamos la equalizacion del histograma
                lkt = fsiv_create_equalization_lookup_table(hist, hold_median);
                // 5. Obtenemos la imagen de salida
                out.at<uchar>(i+radius, j+radius)=lkt.at<uchar>(in.at<uchar>(i+radius, j+radius));
               }
           }
       }
    // 1.2. Equalizamos el histograma
    else{ 
           // 2. Generamos el histograma
           fsiv_compute_histogram(in, hist);
           // 3. Creamos la tabla de equalizacion lookup
           lkt = fsiv_create_equalization_lookup_table(hist, hold_median);
           // 4. Aplicamos lookup
           fsiv_apply_lookup_table(in, lkt, out);
       }
     return out;
}

# color_balance

/**
 * @brief Scale the color of an image so an input color is transformed into an output color.
 * @param in is the image to be rescaled.
 * @param from is the input color.
 * @param to is the output color.
 * @return the color rescaled image.
 * @pre in.type()==CV_8UC3
 * @warning A BGR color space is assumed for the input image.
 */
 
cv::Mat fsiv_color_rescaling(const cv::Mat& in, const cv::Scalar& from, const cv::Scalar& to)
{

    cv::Mat out;
    cv::Scalar rescaling;
    // 1. Obtenemos el reescalado de la imagen
    cv::divide(to, from, rescaling);
    // 2. Reescalamos la imagen
    out=in.mul(rescaling);
    return out;
}

/**
 * @brief Apply a "white patch" color balance operation to the image.
 * @arg[in] in is the imput image.
 * @return the color balanced image.
 * @pre in.type()==CV_8UC3
 * @warning A BGR color space is assumed for the input image.
 */
 
 cv::Mat fsiv_wp_color_balance(cv::Mat const& in)
{
    
    cv::Mat out;
    double min, max;
    cv::Point minp, maxp;
    // 1. Imagen de bgr a gray
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    // 2. Obtenemos el minimo y maximo pixel de la imagen
    cv::minMaxLoc(out , &min, &max, &minp, &maxp);
    // 3. Obtenemos el scalar blanco
    cv::Scalar white = in.at<cv::Vec3b>(maxp);
    // 4. Reescalamos el color blanco de la imagen
    out = fsiv_color_rescaling(in, white, cv::Scalar(255,255,255));
    return out;
}

/**
 * @brief Apply a "gray world" color balance operation to the image.
 * @arg[in] in is the imput image.
 * @return the color balanced image.
 * @pre in.type()==CV_8UC3
 * @warning A BGR color space is assumed for the input image.
 */
 
 cv::Mat fsiv_gw_color_balance(cv::Mat const& in)
{
    
    // 1. Obtenemos el scalar del color gris de la imagen
    cv::Scalar grey = cv::mean(in);
    // 2. Reescalamos el color gris de la imagen
    cv::Mat out = fsiv_color_rescaling(in, grey, cv::Scalar(128,128,128));
    return out;
}

/**
 * @brief Apply a general color balance operation to the image.
 * @arg[in] in is the imput image.
 * @arg[in] p is the percentage of brightest points used to calculate the color correction factor.
 * @return the color balanced image.
 * @pre in.type()==CV_8UC3
 * @pre 0.0 < p < 100.0
 * @warning A BGR color space is assumed for the input image.
 */
 
 cv::Mat fsiv_color_balance(cv::Mat const& in, float p)
{
    
    cv::Mat out, hist;
    // 1. Transformamos la imagen de bgr a gray
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    // 2. Calculamos el histograma de la imagen
    int histSize[]= {256}, canales[] = {0};
    float intensityRanges[] = {0, 256};
    const float* ranges[] = {intensityRanges};
    cv::calcHist(&out, 1, canales, cv::Mat(), hist, 1, histSize, ranges, true, false);
    // 3. Normalizamos el histograma
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1, -1, cv::Mat());
    // 4. Reescalamos el color de la imagen
    out = fsiv_color_rescaling(in, cv::mean(in, out), cv::Scalar(255, 255, 255));
    return out;
}

# usm_enhance

/**
 * @brief Return a box filter.
 * @arg[in] r is the filter's radius.
 * @return the filter.
 * @pre r>0;
 * @post ret_v.type()==CV_32FC1
 * @post retV.rows==retV.cols==2*r+1
 * @post (abs(cv::sum(retV)-1.0)<1.0e-6
 */
 
 cv::Mat fsiv_create_box_filter(const int r)
{
    
    // 1. Generamos el box filter
    cv::Mat ret_v = cv::Mat::ones(2*r+1, 2*r+1, CV_32F) / pow(2*r+1,2);
    return ret_v;
}

/**
 * @brief Return a Gaussian filter.
 * @arg[in] r is the filter's radius.
 * @return the filter.
 * @pre r>0;
 * @post ret_v.type()==CV_32FC1
 * @post retV.rows==retV.cols==2*r+1
 * @post (abs(cv::sum(retV)-1.0)<1.0e-6
 */
 
 cv::Mat fsiv_create_gaussian_filter(const int r)
{
    
    // Remenber 6*sigma is approx 99,73% of the distribution
    // 1. Aplicamos la formula gausiano: (1.0/(2.0*M_PI*pow(2*r+1/6.0,2)) * exp(-(pow(i-r,2)+pow(j-r,2))/(2.0*pow(2*r+1/6.0,2))))
    float size = 2*r+1, a= 1.0/(2.0*M_PI*pow(size/6.0,2));

    cv::Mat ret_v = cv::Mat(size, size, CV_32FC1);
        
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float b = -(pow(i-r,2)+pow(j-r,2))/(2.0*pow(size/6.0,2));
            ret_v.at<float>(i, j) = a * exp(b);
            }
        }
    // 2. Calculamos el filtro gausiano
    ret_v /= (cv::sum(ret_v));
    return ret_v;
}

/**
 * @brief Expand an image with zero padding.
 * @warning the code can't use the interface cv::copyMakeborder().
 * @arg[in] in is the input image.
 * @arg[in] r is the window's radius to expand.
 * @return the expanded image.
 * @pre !in.empty()
 * @pre r>0
 * @post retV.type()==in.type()
 * @post retV.rows==in.rows+2*r
 * @post retV.cols==in.cols+2*r
 */
 
 cv::Mat fsiv_fill_expansion(cv::Mat const& in, const int r)
{
    
    cv::Mat ret_v;
    // 1.1. Opcion CV_32FC1
    if(in.type() == CV_32FC1) ret_v = cv::Mat::zeros(in.rows+2*r, in.cols+2*r, CV_32FC1);
    // 1.2. Opcion CV_8UC1
    else ret_v = cv::Mat::zeros(in.rows+2*r, in.cols+2*r, CV_8UC1);
     // 2. Expandimos la imagen
    in.copyTo(ret_v(cv::Rect(r,r, in.cols, in.rows)));
    return ret_v;
}

/**
 * @brief Circular expansion of an image.
 * @warning the code can't use the interface cv::copyMakeborder().
 * @arg[in] in is the input image.
 * @arg[in] r is the window's radius to expand.
 * @return the expanded image.
 * @pre !in.empty()
 * @pre r>0
 * @post retV.type()==in.type()
 * @post retV.rows==in.rows+2*r
 * @post retV.cols==in.cols+2*r
 */
 
 cv::Mat fsiv_circular_expansion(cv::Mat const& in, const int r)
{
    
    // 1. Aplicamos una fill expansion de la imagen original
    ret_v = fsiv_fill_expansion(in, r);
    // 2. Expansion del rectangulo superior de la imagen de entrada
    in(cv::Rect(0,0,in.cols,r)).copyTo(ret_v(cv::Rect(r, in.rows+r, in.cols,r)));
    // 3. Expansion del rectangulo inferior de la imagen de entrada
    in(cv::Rect(0,in.rows-r,in.cols,r)).copyTo(ret_v(cv::Rect(r, 0, in.cols,r)));
    // 4. Expansion del lado izquierdo de la imagen de entrada
    in(cv::Rect(0,0,r,in.rows)).copyTo(ret_v(cv::Rect(in.cols+r,r,r,in.rows)));
    // 5. Expansion del lado derecho de la imagen de entrada
    in(cv::Rect(in.cols-r,0,r,in.rows)).copyTo(ret_v(cv::Rect(0,r,r,in.rows)));
    // 6. Expansion de la esquina superior izquierda de la imagen de entrada
    in(cv::Rect(0,0,r,r)).copyTo(ret_v(cv::Rect(in.cols+r,in.rows+r,r,r)));
    // 7. Expansion de la esquina superior derecha de la imagen de entrada
    in(cv::Rect(in.cols-r,in.rows-r,r,r)).copyTo(ret_v(cv::Rect(0,0,r,r)));
    // 8. Expansion de la esquina inferior izquierda de la imagen de entrada
    in(cv::Rect(in.cols-r,0,r,r)).copyTo(ret_v(cv::Rect(0,in.rows+r,r,r)));
    // 9. Expansion de la esquina inferior derecha de la imagen de entrada
    in(cv::Rect(0,in.rows-r,r,r)).copyTo(ret_v(cv::Rect(in.cols+r,0,r,r)));
    return ret_v;
}

/**
 * @brief Compute the digital correlation between two images.
 * @warning Code from scracth. Use cv::filter2D() is not allowed.
 * @arg[in] in is the input image.
 * @arg[in] filter is the filter to be applied.
 * @pre !in.empty() && !filter.empty()
 * @pre in.type()==CV_32FC1 && filter.type()==CV_32FC1.
 * @post ret.type()==CV_32FC1
 * @post ret.rows == in.rows-2*(filters.rows/2)
 * @post ret.cols == in.cols-2*(filters.cols/2)
 */
 
 cv::Mat fsiv_filter2D(cv::Mat const& in, cv::Mat const& filter)
{
    
    // 1. Creamos una imagen vacia de salida
    cv::Mat ret_v = cv::Mat::zeros((in.rows-2*(filter.rows/2)), (in.cols-2*(filter.cols/2)), CV_32FC1);
    // 2. Aplicamos el filtro 2D a la imagen
    for (int i = 0; i < ret_v.rows; ++i) {
        for (int j = 0; j < ret_v.cols; ++j){
            ret_v.at<float>(i,j) = sum(filter.mul(in(cv::Rect(j, i, filter.cols, filter.rows)))).val[0];
        }
    }
    return ret_v;
}

/**
 * @brief Combine two images using weigths.
 * @param src1 first image.
 * @param src2 second image.
 * @param a weight for first image.
 * @param b weight for the second image.
 * @return a * src1 + b * src2
 * @pre src1.type()==src2.type()
 * @pre src1.rows==src2.rows
 * @pre src1.cols==src2.cols
 * @post retv.type()==src2.type()
 * @post retv.rows==src2.rows
 * @post retv.cols==src2.cols
 */
 
 cv::Mat fsiv_combine_images(const cv::Mat src1, const cv::Mat src2,double a, double b)
{
    
    // 1. Combinamos las imagenes aplicando la suma ponderada
    cv::Mat ret_v = src1.mul(a) + src2.mul(b);
    return ret_v;
}

cv::Mat fsiv_usm_enhance(cv::Mat  const& in, double g, int r, int filter_type, bool circular, cv::Mat *unsharp_mask)
{
    
    cv::Mat filtro, expansion;
    // 1.1. Opcion box filter -> Creamos el filtro box filter
    if(filter_type == 0) filtro = fsiv_create_box_filter(r);
    // 1.2 Opcion gaussiano -> Creamos el filtro gaussiano
    else filtro = fsiv_create_gaussian_filter(r);
    // 2.1. Opcion no circular -> Aplicamos fill expansion
    if(!circular) expansion = fsiv_fill_expansion(in, r);
    // 2.2. Opcion circular -> Aplicamos la expansion circular
    else expansion = fsiv_circular_expansion(in, r);
    // 3. Aplicamos el filtro 2D
    cv::Mat filtro2D = fsiv_filter2D(expansion, filtro);
    // 4. Opcion unshar mask -> Aplicamos unsharp mask
    if(unsharp_mask) *unsharp_mask = filtro2D;
    // 5. Combinamos las imagenes
    cv::Mat ret_v = fsiv_combine_images(in, filtro2D, 1+g, -g);
    return ret_v;
}

# Sharpening

/**
 * @brief Extend an image centering on the result.
 * @param in the input image.
 * @param new_size is the new geometry.
 * @param ext_type is the type of extension: 0->Padding with zeros. 1->circular.
 * @return the extended image.
 * @pre img.rows<new_size.height
 * @pre img.cols<new_size.width
 * @post ret_v.type()==img.type()
 * @post ret_v.size()==new_size
 */
 
 cv::Mat fsiv_extend_image(const cv::Mat& img, const cv::Size& new_size, int ext_type)
{
    
    cv::Mat out;
     
    // 1.1. Opcion Fill expansion -> Aplicamos fill expansion
    if(ext_type == 0) cv::copyMakeBorder(
                                             img,
                                             out,
                                             (new_size.height-img.rows)/2,
                                             (new_size.height-img.rows)/2,
                                             (new_size.height-img.rows)/2,
                                             (new_size.height-img.rows)/2,
                                             cv::BORDER_CONSTANT);

    // 1.2. Opcion circular -> Aplicamos expansion circular
    else cv::copyMakeBorder(
                              img,
                              out,
                              (new_size.height-img.rows)/2,
                              (new_size.height-img.rows)/2,
                              (new_size.height-img.rows)/2,
                              (new_size.height-img.rows)/2,
                              cv::BORDER_WRAP);
    return out;
}

/**
 * @brief Create a sharpening filter.
 * @param filter_type specify what type of laplacian to use: 0->LAP_4, 1->LAP_8, 2->DoG.
 * @param r1 if filter type is 2 (DoG), r1 is the radius for the first Gaussian filter.
 * @param r2 if filter type is 2 (DoG), r2 is the radius for the second Gaussian filter.
 * @return the filter.
 * @pre filter_type in {0,1,2}
 * @post retval.type()==CV_32FC1
 */
 
 cv::Mat fsiv_create_sharpening_filter(const int filter_type, int r1, int r2)
{
    
    //Hint: use fsiv_extend_image() to extent G[r1].
    // 1. Creamos la matriz del filtro
    cv::Mat filter = cv::Mat::zeros(3, 3, CV_32FC1);
    // 2.1. Diseñamos un filtro lap4 -> [0, -1, 0; -1, 5, -1; 0, -1, 0]
    if(filter_type==0){
        filter.at<float>(0,1) = -1; 
        filter.at<float>(1,0) = -1, filter.at<float>(1,1) = 5, filter.at<float>(1,2) = -1;
        filter.at<float>(2,1) = -1;
    }
    // 2.2. Diseñamos un filtro lap8 -> [-1, -1, -1; -1, 9, -1; -1, -1, -1]
    else if(filter_type==1){
        filter.at<float>(0,0) = -1, filter.at<float>(0,1) = -1, filter.at<float>(0,2) = -1;
        filter.at<float>(1,0) = -1, filter.at<float>(1,1) = 9,  filter.at<float>(1,2) = -1;
        filter.at<float>(2,0) = -1, filter.at<float>(2,1) = -1, filter.at<float>(2,2) = -1;
    }
    // 2.3. Aplicamos giltro DOG -> G[r2]-G[r1]
    else{
        3. Creamos los filtros gaussianos 
        cv::Mat g1 = fsiv_create_gaussian_filter(r1), g2 = fsiv_create_gaussian_filter(r2);        
        // 4. Extendemos la imagen del filtro g1 a g2
        g1 = fsiv_extend_image(g1, g2.size(),0);
        // 5. Obtenemos el filtro DOG
        filter = g1-g2;
        // 6. Añadimos el valor 1 a la posicion central del filtro
        filter.at<float>(g2.rows/2, g2.cols/2) += 1;
    }
    return filter;
}

* @brief Do a sharpeing enhance to an image.
 * @param img is the input image.
 * @param filter_type is the sharpening filter to use: 0->LAP_4, 1->LAP_8, 2->DOG.
 * @param only_luma if the input image is RGB only enhances the luma, else enhances all RGB channels.
 * @param r1 if filter type is DOG, is the radius of first Gaussian filter.
 * @param r2 if filter type is DOG, is the radius of second Gaussian filter.
 * @param circular if it is true, use circular convolution.
 * @return the enahance image.
 * @pre filter_type in {0,1,2}.
 * @pre 0<r1<r2
 */
 
 cv::Mat fsiv_image_sharpening(const cv::Mat& in, int filter_type, bool only_luma, int r1, int r2, bool circular)
{
                
     //Hint: use cv::filter2D.
     //Remenber: if circular, first the input image must be circular extended, and then clip the result.
     cv::Mat out;
     // 1. Creamos el filtro sharpening    
     cv::Mat filter = fsiv_create_sharpening_filter(filter_type, r1, r2);
     // 2. Obtenemos el tamaño de la imagen extendida
     cv::Size new_size (in.cols+(2*r2), in.rows+(2*r2));
     // 3.1. Opcion luma
     if(only_luma && in.channels() == 3){ 
          cv::Mat hsv = in.clone();
          // 4. Convertimos la imagen de BGR a HSV
          cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);
          // 5. Separamos la imagen en canales
          std::vector<cv::Mat> canales;
          cv::split(hsv, canales);
          // 6.1. Opcion circular
          if(circular){
               // 7. Extension circular de la imagen
               cv::Mat circ = fsiv_extend_image(canales[2], new_size, 1);
               // 8. Aplicamos el filtro 2D
               cv::filter2D(circ, circ, -1, filter);
               // 9. Recortamos la imagen
               circ = circ(cv::Rect(r2, r2, in.cols, in.rows));
               // 10. Copiamos la imagen recortada en el canal luma
               circ.copyTo(canales[2]); // Copiar la imagen recortada al canal luma
          }
          // 6.2. Opcion fill expansion
          else{
               // 7. Aplicamos fill expansion
               cv::Mat ext = fsiv_extend_image(canales[2], new_size, 0);
               // 8. Aplicamos el filtro 2D
               cv::filter2D(ext, ext, -1, filter);
               // 9. Recortamos la imagen
               ext = ext(cv::Rect(r2, r2, in.cols, in.rows));
               // 10. Aplicamos el procesamiento al canal luma
               ext.copyTo(canales[2]);
          }
          // 7. Unimos los canales
          cv::merge(canales, hsv);
          // 8. Transformamos la imagen de hsv a bgr
          cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
     }
    // 3.2. Opcion no cluma
    else{
        cv::Mat img, extended;
        // 4.1. Expansion circular
        if(circular) extended = fsiv_extend_image(in,new_size,1);
        // 4.2. fill expansion
        else extended = fsiv_extend_image(in,new_size,0);
        // 5. Aplicamos el filtro 2D
        cv::filter2D(extended, out, -1, filter);
        // 6. Recortamos la imagen
        out = out(cv::Rect(r2,r2,in.cols,in.rows));
    }
    return out;
}

# undishort

/**
 * @brief Generate a 3d point vector with the inner corners of a calibration board.
 * @param board_size is the inner points board geometry (cols x rows).
 * @param square_size is the size of the squares.
 * @return a vector of 3d points with the corners.
 * @post ret_v.size()==(cols*rows)
 */
 
std::vector<cv::Point3f> fsiv_generate_3d_calibration_points(const cv::Size& board_size, float square_size)
{
     
    // 1. Obtenemos los puntos de calibracion 3D -> the first inner point has (1,1) in board coordinates.
    std::vector<cv::Point3f> ret_v;
    for(int i=1; i <= board_size.height; i++){
        for(int j=1; j<= board_size.width; j++){
            ret_v.push_back(cv::Point3f(j*square_size,i*square_size, 0)); // puntos de calibracion 3D
        }
    }
    return ret_v;
}

/**
 * @brief Find a calibration chessboard and compute the refined coordinates of the inner corners.
 * @param img is the image where finding out.
 * @param board_size is the inners board points geometry.
 * @param[out] corner_points save the refined corner coordinates if the board was found.
 * @param wname is its not nullptr, it is the window's name use to show the detected corners.
 * @return true if the board was found.
 * @pre img.type()==CV_8UC3
 * @warning A keyboard press is waited when the image is shown to continue.
 */
 
bool fsiv_find_chessboard_corners(const cv::Mat& img, const cv::Size &board_size,
                                        std::vector<cv::Point2f>& corner_points,const char * wname)
{
    
    // 1. Buscamos las esquinas del tablero
    bool was_found = cv::findChessboardCorners(img,board_size,corner_points);
    // Opcion. esquinas encontradas
    if(was_found){
        cv::Mat grey = img.clone();
        // 2. Convertimos la imagen de bgr a gray
        cv::cvtColor(grey, grey, cv::COLOR_BGR2GRAY);
        // 3. Mejoramos la localizacion de las esquinas
        cv::cornerSubPix(grey, corner_points, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria()); 
    }
    // Opcion. Fichero de salida
    if(wname){
        // 2. Dibujamos las esquinas del tablero
        cv::drawChessboardCorners(img, board_size, corner_points, was_found);
        // 3. Mostramos la imagen
        cv::imshow(wname, img);
    }
    return was_found;
}

/**
 * @brief Calibrate a camara given the a sequence of 3D points and its correspondences in the plane image.
 * @param[in] _2d_points are the sequence of 2d corners detected per view.
 * @param[in] _3d_points are the corresponding 3d points per view.
 * @param[in] camera_size is the camera geometry in pixels.
 * @param[out] camera_matrix is the calibrated camera matrix.
 * @param[out] dist_coeffs is the calibrated distortion coefficients.
 * @param[out] rvects is not null, it will save the rotation matrix of each view.
 * @param[out] tvects is not null, it will save the translation vector of each view.
 * @return the reprojection error of the calibration.
 */
 
 float fsiv_calibrate_camera(const std::vector<std::vector<cv::Point2f>>& _2d_points,
                      const std::vector<std::vector<cv::Point3f>>& _3d_points, const cv::Size &camera_size,
                      cv::Mat& camera_matrix,cv::Mat& dist_coeffs,std::vector<cv::Mat>* rvecs,std::vector<cv::Mat>* tvecs)
{

    float error=0.0;
    // Opcion. Faltan los parametros de la camara -> Calculamos el error de calibracion
    if( (rvecs != nullptr) && (tvecs != nullptr))
        error = cv::calibrateCamera(_3d_points, _2d_points, camera_size, camera_matrix, dist_coeffs, *rvecs, *tvecs);
    
    // Opcion. Se han indicado los parametros de la camara -> Calculamos el error de calibracion
    else{
        std::vector<cv::Mat> rvecs_, tvecs_;
        error = cv::calibrateCamera(_3d_points, _2d_points, camera_size, camera_matrix, dist_coeffs, rvecs_, tvecs_);
    }
    return error;
}


/**
 * @brief Project the 3D Camera Coordinate system on the image.
 * The X axis will be draw in red, the Y axis in green and the Z axis in blue.
 * @param[in,out] img the image where projecting the axes. 
 * @param[in] camera_matrix is the camera matrix.
 * @param[in] dist_coeffs are the distortion coefficients.
 * @param[in] rvec is the rotation vector.
 * @param[in] tvec is the translation vector.
 * @param[in] size is the length in word coordinates of each axis.
 * @param[in] line_width used to draw the axis.
 */
void fsiv_draw_axes(cv::Mat& img, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
               const cv::Mat& rvec, const cv::Mat& tvec,const float size, const int line_width)
{
    
    // 1. Obtenemos los puntos 3D
    std::vector<cv::Point3f> points3d = { 
                                          cv::Point3f(0,0,0),   
                                          cv::Point3f(size, 0, 0), 
                                          cv::Point3f(0, size, 0), 
                                          cv::Point3f(0, 0, -size)
                                          };
    // 2. Proyectamos los puntos 3d
    std::vector<cv::Point2f> points2d;
    cv::projectPoints(points3d, rvec, tvec, camera_matrix, dist_coeffs, points2d); // Realizamos la proyección de los puntos 3D
    // 3. Obtenemos la linea de puntos 2D
    cv::line(img, points2d[0], points2d[1], cv::Scalar(0, 0, 255), line_width);
    cv::line(img, points2d[0], points2d[2], cv::Scalar(0, 255, 0), line_width);
    cv::line(img, points2d[0], points2d[3], cv::Scalar(255, 0, 0), line_width);
}


/**
 * @brief Save the calibration parameters in a file.
 * The labels to save are:
 *
 * image-width (int)
 * image-height (int)
 * error (float/double)
 * camera-matrix (Matrix 3x3 CV_64F)
 * distortion-coefficients (Matrix 1x5 CV_64F)
 * rvec (Matrix 3x1 CV_64F)
 * tvec (Matrix 3x1 CV_64F)
 *
 * @param[in|out] fs is a file storage object to write the data.
 * @param[in] camera_size is the camera geometry in pixels.
 * @param[in] error is the calibration error.
 * @param[in] camera_matrix is the camera matrix.
 * @param[in] dist_coeffs are the distortion coefficients.
 * @param[in] rvec is the rotation vector.
 * @param[in] tvec is the translation vector.
 * @pre fs.isOpened()
 * @post fs.isOpened()
 */
 
 void fsiv_save_calibration_parameters(cv::FileStorage& fs,
                                const cv::Size & camera_size,float error,const cv::Mat& camera_matrix,
                                const cv::Mat& dist_coeffs,const cv::Mat& rvec,const cv::Mat& tvec)
{
    
    // 1. Guardamos en el fichero los parametros de calibracion de la camara
    fs.write("image-width", camera_size.width);
    fs.write("image-height", camera_size.height);
    fs.write("error", error);
    fs.write("camera-matrix", camera_matrix);
    fs.write("distortion-coefficients", dist_coeffs);
    fs.write("rvec", rvec);
    fs.write("tvec", tvec);
    return;
}


/**
 * @brief Compute the pose of a camara giving a view of the board.
 * @param[in] _3dpoints are the WCS 3D points of the board.
 * @param[in] _2dpoints are the refined corners detected.
 * @param[in] camera_matrix is the camera matrix.
 * @param[in] dist_coeffs are the distortion coefficients.
 * @param[out] rvec is the computed rotation vector.
 * @param[out] tvec is the computed translation vector.
 */

void fsiv_compute_camera_pose(const std::vector<cv::Point3f> &_3dpoints,const std::vector<cv::Point2f> &_2dpoints,
                              const cv::Mat& camera_matrix,const cv::Mat& dist_coeffs,cv::Mat& rvec,cv::Mat& tvec)
{

    // 1. Pose de la camara
    cv::solvePnP(_3dpoints, _2dpoints, camera_matrix, dist_coeffs, rvec, tvec);
}

/**
 * @brief Load the calibration parameters from a file.
 * The file will have the following labels:
 *
 * image-width (int)
 * image-height (int)
 * error (float/double)
 * camera-matrix (Matrix 3x3 CV_64F)
 * distortion-coefficients (Matrix 1x5 CV_64F)
 * rvec (Matrix 3x1 CV_64F)
 * tvec (Matrix 3x1 CV_64F)
 *
 * @param[in|out] fs is a file storage object to write the data.
 * @param[out] camera_size is the camera geometry in pixels.
 * @param[out] error is the calibration error.
 * @param[out] camera_matrix is the camera matrix.
 * @param[out] dist_coeffs are the distortion coefficients.
 * @param[out] rvec is the rotation vector.
 * @param[out] tvec is the translation vector.
 * @pre fs.isOpened()
 * @post fs.isOpened()
 */

void fsiv_load_calibration_parameters(cv::FileStorage &fs,cv::Size &camera_size,
                                 float& error,cv::Mat& camera_matrix,cv::Mat& dist_coeffs,cv::Mat& rvec,cv::Mat& tvec)
{
    
    // 1. Cargamos los parametros de calibracion de la camara
    fs["image-width"] >> camera_size.width;
    fs["image-height"] >> camera_size.height;
    fs["error"] >> error;
    fs["camera-matrix"] >> camera_matrix;
    fs["distortion-coefficients"] >> dist_coeffs;
    fs["rvec"] >> rvec;
    fs["tvec"] >> tvec;
    return;
}

/**
 * @brief Correct the len's distorntions of an image.
 * @param[in] input the distorted image.
 * @param[out] output the corrected image.
 * @param[in] camera_matrix is the camera matrix.
 * @param[in] dist_coeffs are the distortion coefficients.
 */

void fsiv_undistort_image(const cv::Mat& input, cv::Mat& output,const cv::Mat& camera_matrix,const cv::Mat& dist_coeffs)
{
    
    //1. Corregimos la distorsion de la lente de la camara -> Hint: use cv::undistort.
    cv::undistort(input, output, camera_matrix, dist_coeffs);   
}

/**
 * @brief Correct the len's distortions from a input video stream.
 * @param[in|out] input is the input distorted video stream.
 * @param[out] output is the corrected output video stream.
 * @param[in] camera_matrix is the camera matrix.
 * @param[in] dist_coeffs are the distortion coefficients.
 * @param[in] interp specifies the interpolation method to use.
 * @param[in] input_wname if it is not nullptr, show input frames.
 * @param[in] output_wname if it is not nullptr, show ouput frames.
 * @param[in] fps is the frame per seconds to wait between frames (when frames are shown). Value 0 means dont wait.
 * @pre input.isOpened()
 * @pre output.isOpened()
 * @post input.isOpened()
 * @post output.isOpened()
 */

void fsiv_undistort_video_stream(cv::VideoCapture&input_stream,cv::VideoWriter& output_stream,const cv::Mat& camera_matrix,
                            const cv::Mat& dist_coeffs,const int interp,const char * input_wname,const char * output_wname,
                            double fps)
{

    //Hint: to speed up, first compute the transformation maps 
    // (one time only at the beginning using cv::initUndistortRectifyMap)
    // and then only remap (cv::remap) the input frame with the computed maps.
    
    cv::Mat frame_in, frame_out, map1, map2;
    
    // 1. Obtenemos el tamaño de la imagen
    cv::Size size = cv::Size(input_stream.get(cv::CAP_PROP_FRAME_WIDTH), input_stream.get(cv::CAP_PROP_FRAME_HEIGHT));
    // 2. Obtenemos la matriz de transformación    
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), camera_matrix, size, CV_32FC1, map1, map2);
    // 3. Frame de entrada
    input_stream >> frame_in;
    // 4. Iniciamos el frame
    while(!frame_in.empty()){
        double retard = 1000/fps;
        cv::remap(frame_in, frame_out, map1, map2, interp);
        output_stream.write(frame_out);
        cv::imshow(input_wname, frame_in); 
        cv::imshow(output_wname, frame_out);
        cv::waitKey(retard);
        input_stream >> frame_in;
    }
}

# bcksegm

/**
 * @brief Remove segmentation noise using morphological operations.
 * @param img image where removing the noise.
 * @param r is the radius of the structuring element.
 * @pre img.type()==CV_8UC1
 * @pre r>0
 */
 
 void fsiv_remove_segmentation_noise(cv::Mat & img, int r)
{
    
    // 1. Obtenemos la estructura
    cv::Size tam(2*r+1,2*r+1);
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, tam);
    // 2. Removemos la segmentacion del ruido -> Primero cerramos y luego abrimos
    cv::Mat dst;
    cv::morphologyEx(img, dst, cv::MORPH_CLOSE, structuringElement);
    cv::morphologyEx(dst, img, cv::MORPH_OPEN, structuringElement);
}

/**
 * @brief Applies a segmentation method based on image difference
 * @param[in] prevFrame Previous image frame (RGB)
 * @param[in] curFrame  Current image frame (RGB)
 * @param[out] difimg  Single-channel generated mask
 * @param[in] thr Theshold used to decide if a pixel contains enough motion to be considered foreground.
 * @param[in] r  Radius to remove segmentation noise. (r=0 means not remove).
 */
 
 void fsiv_segm_by_dif(const cv::Mat & prevFrame, const cv::Mat & curFrame,cv::Mat & difimg, int thr, int r)
{

    // 1. Imagenes en escala de grises
    cv::Mat previous, cursor;
    cv::cvtColor(prevFrame, previous, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curFrame, cursor, cv::COLOR_BGR2GRAY);
    // 2. Generamos la imagen vacia
    cv::Mat zeros = cv::Mat::zeros(prevFrame.size(), prevFrame.type());
    // 3. Diferencia absoluta entre los cursores
    cv::absdiff(previous, cursor, zeros); 
    // 4. Comparamos con thr
    difimg = zeros >= thr;
    // 5. Eliminamos el ruido
    if(r > 0) fsiv_remove_segmentation_noise(difimg, r);
}

/**
 * @brief Applies a mask to an RGB image
 * @param[in] frame RGB input image.
 * @param[in] mask  Single-channel mask.
 * @param[out] outframe Output RGB frame.
 */
 
 void fsiv_apply_mask(const cv::Mat & frame, const cv::Mat & mask,cv::Mat & outframe)
{

    cv::Mat masked;
    // 1.1. Imagen con 3 canales -> mascara de escala de gray a bgr
    if(frame.channels()==3) // Frame tiene 3 canales
    {
        // 2. Generamos la mascara vacia
        masked = cv::Mat::zeros(frame.size(), CV_8UC1);
        // 3. Transformamos la mascara de gray a bgr
        cv::cvtColor(mask, masked, cv::COLOR_GRAY2BGR);
    }
    // 1.2. Imagen no tiene 3 canales
    else masked = mask;
    // 2. Aplicamos la mascara
    outframe = frame & masked;
}

/**
 * @brief Learns a gaussian background model given an input stream.
 * @param input     RGB input image.
 * @param[out] mean  the mean image.
 * @param[out] variance the variance image.
 * @param[in] num_frames Number of frames used to estimated the model.
 * @param[in] gauss_r is the radius used to gaussian avegaging of input frames.
 * @param[in] wname Window used to show the captured frames (if gived).
 */
 
 bool fsiv_learn_gaussian_model(cv::VideoCapture & input, cv::Mat & mean,cv::Mat & variance,
                                   int num_frames,int gauss_r,const char * wname)
{
    
    bool was_ok = true;
    // Remenber you can compute the variance as: varI = sum_n{I^2}/n - meanI²
    // Hint: convert to input frames to float [0,1]. use cv::accumulate() and cv::accumulateSquare().
    cv::Mat frame;
    int key=0, i=0, size=2*gauss_r+1;
    // 1. Esperamos a pulsar ESC o alcancer el numero de frames
    while(was_ok && key!=27 && i < num_frames)
    {
        // 2. Leemos el frame
        was_ok=input.read(frame);
        // 3.1. Frame leido correctamente
        if(was_ok)
        {
            // 4. Frame con valores flotantes
            frame.convertTo(frame, CV_32F, 1/255.0);
            // 5.1. Aplicamos un filtro gaussiano
            if(gauss_r>0) cv::GaussianBlur(frame, frame, cv::Size(size, size), 0.0);
            // 6.1. Media y varianza vacia -> Calculamos la media y la varianza
            if(mean.empty() || variance.empty())
            {
                mean=frame.clone();
                variance=frame.mul(frame);
            }
            // 6.2. Media y varianza no vacias -> Calculamos la media y la varianza
            else
            {
                cv::accumulate(frame, mean);
                cv::accumulateSquare(frame, variance);
            }
            // 7. Incrementamos i
            i++;
            // 8. Mostramos la imagen
            if(wname) cv::imshow(wname, frame);
        }
    }
    // 9. Calculamos la media y la varianza
    if(was_ok)
    {   
        mean = mean.mul(1.0/i);
        variance=variance.mul(1.0/i) - mean.mul(mean);
    }
    return was_ok;
}

/**
 * @brief Applies a segmentation method based on a Gaussian model of the background.
 * @param[in] frame RGB input image.
 * @param[out] mask  mask image with foreground pixels to 255.
 * @param[in] mean Model's mean of each RGB pixel.
 * @param[in] variance Model's variance of each RGB pixel.
 * @param[in] k define the detection threshold.
 * @param[in] r radius used to remove segmentation noise (value 0 means not remove).
 */
 
 void fsiv_segm_by_gaussian_model(const cv::Mat & frame,cv::Mat & mask,
                            const cv::Mat & mean,const cv::Mat & variance, float k, int r)
{

    //Remenber: a point belongs to the foreground (255) if |mean-I| >= k*stdev
    cv::Mat diff, sqrt;
    cv::absdiff(frame, mean, diff);
    // 1. Obtenemos la desviacion tipica
    cv::sqrt(variance, sqrt);
    sqrt*=k;
    // 2. Mascara
    cv::Mat masked = diff >= sqrt;
    // 3. Dividimos la mascara en canales
    std::vector<cv::Mat> vector;
    cv::split(masked, vector);
    // 4. Disyuncion entre los elementos del vector de canales
    cv::bitwise_or(vector[0], vector[1], mask);
    cv::bitwise_or(mask, vector[2], mask);
    // 5. Si el radio es positivo, eliminamos el ruido de segmentacion
    if(r>0) fsiv_remove_segmentation_noise(mask, r);
}

/**
 * @brief Update the Background Gaussian model.
 * @param[in] frame is current frame image.
 * @param[in] mask is the current segmentation mask.
 * @param[in] frame_count is the number of this frame into the stream.
 * @param[in,out] mean is the current Model's mean to be updated.
 * @param[in,out] variance is the current Model's variance to be updated.
 * @param[in] alpha is the update rate.
 * @param[in] short_term_update_period specifies the short term update period.
 * @param[in] long_term_update_period specifies the long term update period.
 */
 
 void fsiv_update_gaussian_model(const cv::Mat & frame,const cv::Mat & mask,
                           unsigned long frame_count,cv::Mat & mean,cv::Mat & variance,
                           float alpha,unsigned short_term_update_period,unsigned long_term_update_period)
{
    
    //Remember: In the short term updating you must update the model using the background only (not mask).
    //However in the long term updating you must update the model using both background and foreground (without mask).
    //Hint: a period is met when (idx % period) == 0
    //Hint: use accumulateWeighted to update the model.

    // 1. Matriz negativa de la mascara
    cv::Mat negative;
    cv::bitwise_not(mask, negative);
    
    // 2. short term updating
    if(short_term_update_period > 0 && frame_count % short_term_update_period == 0){
        // 3. computes a running average of the frames
        cv::accumulateWeighted(frame, mean, alpha, negative);
        cv::accumulateWeighted(frame.mul(frame), variance, alpha, negative);
    }
    
    else if(long_term_update_period > 0 && frame_count % long_term_update_period == 0){
        // 4. computes a running average of the frames
        cv::accumulateWeighted(frame, mean, alpha);
        cv::accumulateWeighted(frame.mul(frame), variance, alpha);
    }
}

# aug_real

/**
 * @brief Project a 3D Model on the image.
 * @param[in,out] img the image where projecting the axes.
 * @param[in] camera_matrix is the camera matrix.
 * @param[in] dist_coeffs are the distortion coefficients.
 * @param[in] rvec is the rotation vector.
 * @param[in] tvec is the translation vector.
 * @param[in] size is the length in word coordinates of each axis.
 * @pre img.type()=CV_8UC3
 */
 
 void fsiv_draw_3d_model(cv::Mat &img, const cv::Mat& M, const cv::Mat& dist_coeffs,
                   const cv::Mat& rvec, const cv::Mat& tvec,const float size)
{
    
    // 1. Vector de puntos 3D
    std::vector<cv::Point3f> _3d_points = {
        cv::Point3f(0,0,0), cv::Point3f(size,0,0), cv::Point3f(0,size,0), cv::Point3f(size,size,0), 
        cv::Point3f(size/2,size/2,-size/2)};
    // 2. Vector de puntos 2D
    std::vector<cv::Point2f> _2d_points(_3d_points.size());
    // 3. Proyectamos los puntos 3D
    cv::projectPoints(_3d_points, rvec, tvec, M, dist_coeffs, _2d_points);
    // 4. Unimos los puntos 2D
    cv::line(img, _2d_points[0], _2d_points[1], cv::Scalar(0, 0, 255), 3);
    cv::line(img, _2d_points[1], _2d_points[3], cv::Scalar(0, 255, 0), 3);
    cv::line(img, _2d_points[3], _2d_points[2], cv::Scalar(255, 0, 0), 3);
    cv::line(img, _2d_points[2], _2d_points[0], cv::Scalar(0, 0, 255), 3);
    cv::line(img, _2d_points[0], _2d_points[4], cv::Scalar(0, 255, 0), 3);
    cv::line(img, _2d_points[1], _2d_points[4], cv::Scalar(255, 0, 0), 3);
    cv::line(img, _2d_points[2], _2d_points[4], cv::Scalar(0, 0, 255), 3);
    cv::line(img, _2d_points[3], _2d_points[4], cv::Scalar(0, 255, 0), 3);
}


/**
 * @brief Project input image on the output using the homography of the calibration board on the image plane.
 * @arg[in] input is the image to be projected.
 * @arg[in|out] is the output image.
 * @arg[in] board_size is the inner board gemometry.
 * @arg[in] _2dpoints are the image coordinates of the board corners.
 * @pre input.type()==CV_8UC3.
 * @pre output.type()==CV_8UC3.
 */
 
 void fsiv_project_image(const cv::Mat& input, cv::Mat& output,const cv::Size& board_size,
                   const std::vector<cv::Point2f>& _2dpoints)
{
    
    // 1. Creamos la máscara
    cv::Mat mask = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
    // 2. Esquinas del tablero
    std::vector<cv::Point2f> _2dpoints_ = {
               _2dpoints[0], 
               _2dpoints[board_size.width-1], 
               _2dpoints[board_size.height * board_size.width-1], 
               _2dpoints[(board_size.height-1) * board_size.width]
    };
    std::vector<cv::Point> corners = {_2dpoints_[0], _2dpoints_[1], _2dpoints_[2], _2dpoints_[3]};
    // 3. Rellenamos la máscara
    cv::fillConvexPoly(mask, corners, cv::Scalar(255, 255, 255));
    // 4. Obtenemos la transformación de perspectiva
    std::vector<cv::Point2f> _2dpoints_i = {
               cv::Point2f(0, 0), 
               cv::Point2f(input.cols-1, 0), 
               cv::Point2f(input.cols-1, input.rows-1), 
               cv::Point2f(0, input.rows-1)
    };
    cv::Mat transform = cv::getPerspectiveTransform(_2dpoints_i, _2dpoints_);
    // 5. Aplicamos la transformación de perspectiva
    cv::Mat out = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
    cv::warpPerspective(input, out, transform, output.size());
    // 6. Copiamos la imagen
    out.copyTo(output, mask);
}
 
 # histapathology
 
 /**
 * @brief Create a KNN classifier.
 *
 * @param K is the number of NN neighbours used to classify a sample.
 * @return an instance of the classifier.
 */
 
 cv::Ptr<cv::ml::StatModel> fsiv_create_knn_classifier(int K)
{
    
    cv::Ptr<cv::ml::KNearest> knn;
    // 1. Creamos el KNN
    knn = cv::ml::KNearest::create();
    // 2. Parametro por defecto KNN
    knn->setDefaultK(K);
    return knn;
}

/**
 * @brief Create a SVM classifier.
 *
 * @param Kernel is the kernel type @see cv::ml::SVM::KernelTypes
 * @param C is the SVM's C parameter.
 * @param degree is the degree when kernel type is POLY
 * @param gamma is the gamma exponent when kernel type is RBF.
 * @return an instance of the classifier
 */
 
 cv::Ptr<cv::ml::StatModel> fsiv_create_svm_classifier(int Kernel,double C,double degree,double gamma)
{
    
    // 1. Creamos el algoritmo svm
    cv::Ptr<cv::ml::SVM> svm;
    svm = cv::ml::SVM::create();
    // 2. Introducimos los parametros svm
    svm->setKernel(Kernel);
    svm->setC(C);
    svm->setDegree(degree);
    svm->setGamma(gamma);
    return svm;
}

/**
 * @brief Create a Random Trees classifier.
 *
 * @param V is the number of features used by node. Value 0 means not set this value.
 * @param T is the maximum number of generated trees.
 * @param E is minimun the OOB error allowed.
 * @return an instance of the classifier
 */
 
 cv::Ptr<cv::ml::StatModel> fsiv_create_rtrees_classifier(int V,int T,double E)
{
    
    // 1. Creamos el arbol de decision
    cv::Ptr<cv::ml::RTrees> rtrees;
    rtrees = cv::ml::RTrees::create();
    // 2. Introducimos los parametros del arbol de decision
    rtrees->setActiveVarCount(V);
    rtrees->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, T, E));
    return rtrees;
}

/**
 * @brief Train a classifier.
 *
 * @param clsf the instance of the classifier to be trainned.
 * @param samples are the input samples.
 * @param labels are the input labels.
 * @param flags are the train flags.
 * @pre clfs != nullptr
 * @post clfs->isTrained()
 */
 
 void fsiv_train_classifier(cv::Ptr<cv::ml::StatModel> &clsf,const cv::Mat &samples, const cv::Mat &labels,int flags)
{
    
    // 1. Obtenemos los datos de entrenamiento
    cv::Ptr<cv::ml::TrainData> data =  cv::ml::TrainData::create(samples,cv::ml::ROW_SAMPLE, labels);
    // 2. Entrenamos el clasificador
    clsf->train(cv::ml::TrainData::create(samples,cv::ml::ROW_SAMPLE, labels));
}

/**
 * @brief Make predictions using a trained classifier.
 *
 * @param clsf an instance of the classifier to be used.
 * @param samples the samples to predict the labels.
 * @param predictions the labels predicted.
 * @pre clsf->isTrained()
 * @post predictions.depth()=CV_32S
 * @post predictions.rows == samples.rows.
 */
 
 void fsiv_make_predictions(cv::Ptr<cv::ml::StatModel> &clsf,const cv::Mat &samples, cv::Mat &predictions)
{
    
    // 1. Predicciones del clasificador
    clsf->predict(samples, predictions);
    // 2. Transformamos las predicciones a CV_32S
    predictions.convertTo(predictions, CV_32S);
}

/**
 * @brief Load a knn classifier's model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 * @post ret_v != nullptr
 */
 
 cv::Ptr<cv::ml::StatModel> fsiv_load_knn_classifier_model(const std::string &model_fname)
{
    
    // 1. Leemos los datos del clasificador knn 
    cv::Ptr<cv::ml::StatModel> clsf = cv::Algorithm::load<cv::ml::KNearest>(model_fname);
    return clsf;
}

/**
 * @brief Load a svm classifier's model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 * @post ret_v != nullptr
 */
 
 cv::Ptr<cv::ml::StatModel>
fsiv_load_svm_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // TODO: load a SVM classifier.
    // Hint: use the generic interface cv::Algorithm::load< classifier_type >
    clsf = cv::Algorithm::load<cv::ml::SVM>(model_fname);
    //

    CV_Assert(clsf != nullptr);
    return clsf;
}

/**
 * @brief Load a rtrees classifier's model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 * @post ret_v != nullptr
 */
 
 cv::Ptr<cv::ml::StatModel>
fsiv_load_rtrees_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // TODO: load a RTrees classifier.
    // Hint: use the generic interface cv::Algorithm::load< classifier_type >
    clsf = cv::Algorithm::load<cv::ml::RTrees>(model_fname);
    //

    CV_Assert(clsf != nullptr);
    return clsf;
}

/**
 * @brief Load a classifier model from a file storage.
 * 
 * @param[in] model_fname  is the file storage pathname.
 * @param[out] clsf_id is the loaded classifier id.
 * @return the classifier instance. 
 */
 
 cv::Ptr<cv::ml::StatModel>
load_classifier_model(const std::string &model_fname, int& clsf_id)
{
    cv::FileStorage f;
    cv::Ptr<cv::ml::StatModel> clsf;

    f.open(model_fname, cv::FileStorage::READ);
    auto node = f["fsiv_classifier_type"];
    if (node.empty() || !node.isInt())
        throw std::runtime_error("Could not find 'fsiv_classifier_type' "
                                 "label in model file");    
    node >> clsf_id;
    f.release();
    if (clsf_id == 0)
        clsf = fsiv_load_knn_classifier_model(model_fname);
    else if (clsf_id == 1)
        clsf = fsiv_load_svm_classifier_model(model_fname);
    else if (clsf_id == 2)
        clsf = fsiv_load_rtrees_classifier_model(model_fname);
    else
        throw std::runtime_error("Unknown classifier id: " +
                                 std::to_string(clsf_id));

    CV_Assert(clsf != nullptr);
    return clsf;
}

/**
 * @brief Compute the confussion matrix.
 * 
 * Is a matrix where the rows are the ground-truth labels and the columns are
 * the predicted labels.
 * 
 * @param true_labels are supervised labels.
 * @param predicted_labels are the predicted labels.
 * @param n_categories is the number of categories.
 * @return the confussion matrix.
 */
 
 cv::Mat
fsiv_compute_confusion_matrix(const cv::Mat &true_labels,
                              const cv::Mat &predicted_labels,
                              int n_categories)
{
    CV_Assert(true_labels.rows == predicted_labels.rows);
    CV_Assert(true_labels.type() == CV_32SC1);
    CV_Assert(predicted_labels.type() == CV_32SC1);
    cv::Mat cmat = cv::Mat::zeros(n_categories, n_categories, CV_32F);

    //TODO: Compute the confussion matrix.
    //Remenber: Rows are the Ground Truth. Cols are the predictions.
    
    for(int i = 0; i < true_labels.rows; i++){
        int row = true_labels.at<int>(i);
        if(row < 0){
            row = 0;
        }
        int col = predicted_labels.at<int>(i);
        if(col < 0){
            col = 0;
        }
        cmat.at<float>(row, col) += 1;
    }

    //
    CV_Assert(std::abs(cv::sum(cmat)[0] - static_cast<double>(true_labels.rows)) <=
              1.0e-6);
    return cmat;
}

/**
 * @brief Compute the accuracy metrix.
 * 
 * @param cmat is the confussion matrix.
 * @return the accuracy.
 */
 
 float fsiv_compute_accuracy(const cv::Mat &cmat)
{
    CV_Assert(!cmat.empty() && cmat.type() == CV_32FC1);
    CV_Assert(cmat.rows == cmat.cols && cmat.rows > 1);

    float acc = 0.0;

    //TODO: compute the accuracy.
    //Hint: the accuracy is the rate of correct classifications
    //  to the total.
    //Remenber: avoid zero divisions!!.
    // total = np.sum(cmat)
    float total = 0;
    for(int i = 0; i < cmat.rows; i++){
        for(int j = 0; j < cmat.cols; j++){
            total += cmat.at<float>(i, j);
        }
    }
    // if total > 0.0
    if(total > 0.0){
        // diag = np.sum(np.diag(cmat))
        float diag = 0;
        for(int i = 0; i < cmat.rows; i++){
            diag += cmat.at<float>(i, i);
        }
        // acc = diag / total
        acc = diag / total;
    }
    //
    CV_Assert(acc >= 0.0f && acc <= 1.0f);
    return acc;
}

/**
 * @brief Compute the mean recognition rate.
 * 
 * @param rr are the recognition rate per category.
 * @return the mean recognition rate.
 */
 
 float fsiv_compute_mean_recognition_rate(const std::vector<float> &rr)
{
    float m_rr = 0.0;
    //TODO
    //Remenber: the MRR is the mean value of the recognition rates.
    // total = np.sum(rr)
    float total = 0;
    for(int i = 0; i < rr.size(); i++){
        total += rr[i];
    }
    // if total > 0.0
    if(total > 0.0){
        // m_rr = total / len(rr)
        m_rr = total / rr.size();
    }
    
    //
    return m_rr;
}

/**
 * @brief Output model metrics.
 * 
 * @param gt_labels are supervised labels.
 * @param predicted_labels are predicted labels.
 * @param categories is a vector with the categories names.
 * @param out is the stream to print out.
 */
 
 void print_model_metrics(const cv::Mat &gt_labels,
                         const cv::Mat &predicted_labels,
                         const std::vector<std::string>& categories,
                         std::ostream &out)
{
    cv::Mat cmat = fsiv_compute_confusion_matrix(gt_labels, predicted_labels,
                                                 categories.size());
    float acc = fsiv_compute_accuracy(cmat);
    std::vector<float> rr = fsiv_compute_recognition_rates(cmat);
    float m_rr = fsiv_compute_mean_recognition_rate(rr);
    out << "#########################" << std::endl;
    out << "Model metrics:         " << std::endl;
    out << std::endl;
    out << "Recognition rate per class:" << std::endl;
    for (size_t i = 0; i < rr.size(); ++i)
        out << std::setw(20) << std::setfill(' ')
            << categories[i]
            << ": " << rr[i] << std::endl;
    out << std::endl;
    out << "Mean recognition rate: " << m_rr << std::endl;
    out << "Accuracy: " << acc << std::endl;
    

}

/**
 * @brief compute the recogniton rate per category.
 * 
 * @param cmat is the confussion matrix.
 * @return a vector with the recognition rate per category.
 */
 
 std::vector<float>
fsiv_compute_recognition_rates(const cv::Mat &cmat)
{
    CV_Assert(!cmat.empty() && cmat.type() == CV_32FC1);
    CV_Assert(cmat.rows == cmat.cols);
    std::vector<float> RR(cmat.rows);

    for (int category = 0; category < cmat.rows; ++category)
    {
        RR[category] = 0.0;

        //TODO: compute the recognition rate (RR) for the category.
        //Avoid zero divisions!!.
        //  to the total of samples of the category.
        float total = 0;
        for(int i = 0; i < cmat.rows; i++){
            total += cmat.at<float>(category, i);
        }
        if(total != 0){
            RR[category] = cmat.at<float>(category, category) / total;
        }

        //
        CV_Assert(RR[category] >= 0.0f && RR[category] <= 1.0f);
    }
    return RR;
}
