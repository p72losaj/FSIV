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
     
    cv::Mat out;

    cv::cvtColor(img,out,cv::COLOR_GRAY2BGR);  // Convertimos a RGB
    
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
    
    cv::cvtColor(img,out,cv::COLOR_BGR2GRAY); // Convertimos a gray    
    
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
     
     cv::Mat mask = cv::Mat::zeros(img_height,img_width, type); // mascara 
     
     cv::rectangle(mask, cv::Point(x,y), cv::Point(x+rect_width,y+rect_height), cv::Scalar(255,255,255), cv::FILLED); // rectangulo
     
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
    
    cv::Mat mask = cv::Mat::zeros(img_height, img_width, type); // mascara
    
    cv::circle(mask, cv::Point(x,y), radius, cv::Scalar(255,255,255), cv::FILLED); // circulo
    
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
    
    cv::Mat mask = cv::Mat::zeros(img_height, img_width, type); // Mascara
    
    std::vector< std::vector<cv::Point> > polys; // vector de puntos
    
    polys.push_back(points); // guardamos los puntos
    
    cv::fillPoly(mask, polys, cv::Scalar(255,255,255)); // poligono
    
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
    
    cv::bitwise_and(foreground, mask,foreground); // Primer plano
    
    cv::bitwise_not(mask, mask); // Negamos la mascara
    
    cv::bitwise_and(background, mask, background); // Fondo de la imagen
    
    cv::bitwise_or(foreground, background,output); // combinar imagenes
    
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
     
    img.convertTo(out,CV_32F, 1.0/255.0); // imagen con valores flotante
    
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
    
    img.convertTo(out, CV_8U, 255.0); // imagen con valores byte
    
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
    
    cv::cvtColor(img, out, cv::COLOR_BGR2HSV); // imagen hsv
    
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
    
    cv::cvtColor(img,out,cv::COLOR_HSV2BGR); // imagen bgr
    
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
    
    std::vector<cv::Mat> canales; // vector de canales
    
    convert_image_byte_to_float(in,out); // imagen de salida en valores flotante
    
    if(only_luma && in.channels()==3) convert_bgr_to_hsv(out,out); // Canal luma -> imagen salida en hsv
    
    cv::split(out,canales); // dividimos la imagen de salida en canales
    
    if(only_luma && in.channels() == 3){ // canal luma -> canal_hsv = canal_hsv^g * c + b
        cv::pow(canales[2], gamma, canales[2]); // canal_hsv = canal_hsv^g
        canales[2] = contrast * canales[2] + brightness; // canal_hsv = canal_hsv*c+b
    }
    
    else{ // sin canal luma -> canales^g * c + b
        for(size_t i=0; i<canales.size(); i++){ // recorremos los canales
            cv::pow(canales[i], gamma, canales[i]); // canal = canal^g
            canales[i] = contrast * canales[i] + brightness; // canal = canal*c+b
        }
    }
    
    cv::merge(canales, out); // unimos los canales -> formar imagen de salida
    
    if(only_luma && in.channels() == 3) convert_hsv_to_bgr(out,out); // canal luma -> imagen salida en bgr
    
    convert_image_float_to_byte(out,out); // imagen salida en byte
    
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
    
    int histSize[]= {256}, channels[] = {0};
    
    float intensityRanges[] = {0, 256};
    
    const float* ranges[] = {intensityRanges};
    
    cv::calcHist(&in, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false); // calculamos el histograma
    
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
    
    cv::normalize(hist,hist,1,0, cv::NORM_L1, -1, cv::Mat()); // normalizamos el histograma
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
    
    for(size_t i=0; i<hist.rows; i++) hist.at<float>(i) += hist.at<float>(i-1); // histograma acumulado
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
    
    cv::Mat lkt = hist.clone(); // clonamos el histograma
    
    fsiv_normalize_histogram(lkt); // normalizamos el histograma
    
    fsiv_accumulate_histogram(lkt); // histograma acumulado
    
    if(hold_median){ // hold_median = true
        
        int median = 0;
        
        float position=0;
        
        while(position < 0.5 && median < 256){ // Posicion media del histograma
            median++;
            position = lkt.at<float>(median);
        }
        
        if(0<median<255){
            
            float medianf = median;
            
            cv::Range first_half(0,median), second_half(median,256); // Mitades del histograma
            
            // Primera mitad del histograma
            lkt(first_half, cv::Range::all()) /= position ; 
            lkt(first_half, cv::Range::all()) *= median;
            // Segunda mitad del histograma
            lkt(second_half, cv::Range::all()) -= position;
            lkt(second_half, cv::Range::all()) /= (1-position) ;
            lkt(second_half, cv::Range::all())*=255.0-median;
            lkt(second_half, cv::Range::all())+=medianf;
        }
        lkt.convertTo(lkt, CV_8UC1); // Imagen en CV_8UC1
    }
    
    else lkt.convertTo(lkt, CV_8UC1, 255.0); // Imagen en CV_8UC1
    
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

    cv::LUT(in,lkt,out); // lookup table
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
    out = in.clone(); // copiamos la imagen de entrada
    
    if(radius > 0){ // Aplicamos equualizacion local
        for(int i = 0; i <= in.rows - (2*radius+1); i++){
            for(int j = 0; j <= in.cols - (2*radius+1); j++){
                cv::Mat ventana =  in(cv::Rect(j, i, 2*radius+1, 2*radius+1)); // ventana de radio r
                fsiv_compute_histogram(ventana, hist); // computamos el histograma
                lkt = fsiv_create_equalization_lookup_table(hist, hold_median); // equalizamos el histograma
                out.at<uchar>(i+radius, j+radius)=lkt.at<uchar>(in.at<uchar>(i+radius, j+radius)); // obtenemos la imagen de salida
               }
           }
       }
    
    else{ // Equalizamos el histograma
           fsiv_compute_histogram(in, hist); // generamos el histograma
           lkt = fsiv_create_equalization_lookup_table(hist, hold_median); // equalizamos el histograma
           fsiv_apply_lookup_table(in, lkt, out); // aplicamos lookup table
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
    
    cv::divide(to, from, rescaling); // Obtenemos el reescalado de la imagen
    
    out=in.mul(rescaling); // aplicamos el reescalado de la imagen
    
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
    
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY); // imagen en gray
    
    cv::minMaxLoc(out , &min, &max, &minp, &maxp); // Localizamos el minimo y maximo de la imagen
    
    cv::Scalar white = in.at<cv::Vec3b>(maxp); // Obtenemos el escalar del blanco de la imagen
    
    out = fsiv_color_rescaling(in, white, cv::Scalar(255,255,255)); // Reescalamos la imagen
    
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
    
    cv::Scalar grey = cv::mean(in); // Escalar gris de la imagen
    
    cv::Mat out = fsiv_color_rescaling(in, grey, cv::Scalar(128,128,128)); // reescalamos la imagen
    
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
    int histSize[]= {256}, canales[] = {0};
    float intensityRanges[] = {0, 256};
    const float* ranges[] = {intensityRanges};
    
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY); // imagen de salida en gray
    
    cv::calcHist(&out, 1, canales, cv::Mat(), hist, 1, histSize, ranges, true, false); // Calculate histogram
    
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1, -1, cv::Mat()); // Normalize histogram
    
    out = fsiv_color_rescaling(in, cv::mean(in, out), cv::Scalar(255, 255, 255)); // Reescalar histograma
    
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
    
    cv::Mat ret_v = cv::Mat::ones(2*r+1, 2*r+1, CV_32F) / pow(2*r+1,2); // box filter
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
    // gaussiano = media(ret_v) = (1.0/(2.0*M_PI*pow(2*r+1/6.0,2)) * exp(-(pow(i-r,2)+pow(j-r,2))/(2.0*pow(2*r+1/6.0,2)))) / sumatorio(ret_v)

    float size = 2*r+1, a= 1.0/(2.0*M_PI*pow(size/6.0,2));

    cv::Mat ret_v = cv::Mat(size, size, CV_32FC1);
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float b = -(pow(i-r,2)+pow(j-r,2))/(2.0*pow(size/6.0,2));
            ret_v.at<float>(i, j) = a * exp(b);
            }
        }

    ret_v /= (cv::sum(ret_v)); // filtro gaussiano
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

    if(in.type() == CV_32FC1) ret_v = cv::Mat::zeros(in.rows+2*r, in.cols+2*r, CV_32FC1);

    else ret_v = cv::Mat::zeros(in.rows+2*r, in.cols+2*r, CV_8UC1);

    in.copyTo(ret_v(cv::Rect(r,r, in.cols, in.rows))); // expansion de la imagen original
    
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
    
    // imagen(x,y,w,h)
    
    
    ret_v = fsiv_fill_expansion(in, r); // Expansion de la imagen original
    
    // Expansion del rectangulo superior de la imagen de entrada
    in(cv::Rect(0,0,in.cols,r)).copyTo(ret_v(cv::Rect(r, in.rows+r, in.cols,r)));
    
    // Expansion del rectangulo inferior de la imagen de entrada
    in(cv::Rect(0,in.rows-r,in.cols,r)).copyTo(ret_v(cv::Rect(r, 0, in.cols,r)));
    
    // Expansion del lado izquierdo de la imagen de entrada
    in(cv::Rect(0,0,r,in.rows)).copyTo(ret_v(cv::Rect(in.cols+r,r,r,in.rows)));
    
    // Expansion del lado derecho de la imagen de entrada
    in(cv::Rect(in.cols-r,0,r,in.rows)).copyTo(ret_v(cv::Rect(0,r,r,in.rows)));
    
    // Expansion de la esquina superior izquierda de la imagen de entrada
    in(cv::Rect(0,0,r,r)).copyTo(ret_v(cv::Rect(in.cols+r,in.rows+r,r,r)));
    
    // Expansion de la esquina superior derecha de la imagen de entrada
    in(cv::Rect(in.cols-r,in.rows-r,r,r)).copyTo(ret_v(cv::Rect(0,0,r,r)));
    
    // Expansion de la esquina inferior izquierda de la imagen de entrada
    in(cv::Rect(in.cols-r,0,r,r)).copyTo(ret_v(cv::Rect(0,in.rows+r,r,r)));
    
    // Expansion de la esquina inferior derecha de la imagen de entrada
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
    
    cv::Mat ret_v = cv::Mat::zeros((in.rows-2*(filter.rows/2)), (in.cols-2*(filter.cols/2)), CV_32FC1); // Imagen de salida
    
    for (int i = 0; i < ret_v.rows; ++i) {
        for (int j = 0; j < ret_v.cols; ++j){
            ret_v.at<float>(i,j) = sum(filter.mul(in(cv::Rect(j, i, filter.cols, filter.rows)))).val[0]; // Aplicamos el filtro 2D
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
    
    cv::Mat ret_v = src1.mul(a) + src2.mul(b); // Suma ponderada de las imagenes src1 y src2
    
    return ret_v;
}

cv::Mat fsiv_usm_enhance(cv::Mat  const& in, double g, int r, int filter_type, bool circular, cv::Mat *unsharp_mask)
{
    
    //Hint: use your own functions fsiv_xxxx

    cv::Mat filtro, expansion;
    
    if(filter_type == 0) filtro = fsiv_create_box_filter(r); // box filter
    
    else filtro = fsiv_create_gaussian_filter(r); // filtro gaussiano
        
    if(!circular) expansion = fsiv_fill_expansion(in, r); // expansion no circular
    
    else expansion = fsiv_circular_expansion(in, r); // expansion circular
    
    cv::Mat filtro2D = fsiv_filter2D(expansion, filtro); // filtro 2D
    
    if(unsharp_mask) *unsharp_mask = filtro2D; // unsharp mask
    
    cv::Mat ret_v = fsiv_combine_images(in, filtro2D, 1+g, -g); // suma ponderada de las imagenes in y filtro2D

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
     
    // Fill expansion
    if(ext_type == 0) cv::copyMakeBorder(img,out,(new_size.height-img.rows)/2,(new_size.height-img.rows)/2
                        ,(new_size.height-img.rows)/2,(new_size.height-img.rows)/2,cv::BORDER_CONSTANT);

    // Expansion circular
    else cv::copyMakeBorder(img,out,(new_size.height-img.rows)/2,(new_size.height-img.rows)/2,
                            (new_size.height-img.rows)/2,(new_size.height-img.rows)/2,cv::BORDER_WRAP);
    
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
    
    cv::Mat filter = cv::Mat::zeros(3, 3, CV_32FC1); // Matriz de filtro

    if(filter_type==0){ // [0, -1, 0; -1, 5, -1; 0, -1, 0]
        filter.at<float>(0,1) = -1; 
        filter.at<float>(1,0) = -1, filter.at<float>(1,1) = 5, filter.at<float>(1,2) = -1;
        filter.at<float>(2,1) = -1;
    }

    else if(filter_type==1){ // [-1, -1, -1; -1, 9, -1; -1, -1, -1]
        filter.at<float>(0,0) = -1, filter.at<float>(0,1) = -1, filter.at<float>(0,2) = -1;
        filter.at<float>(1,0) = -1, filter.at<float>(1,1) = 9,  filter.at<float>(1,2) = -1;
        filter.at<float>(2,0) = -1, filter.at<float>(2,1) = -1, filter.at<float>(2,2) = -1;
    }

    else{ // DoG = G[r2]-G[r1]        
        cv::Mat g1 = fsiv_create_gaussian_filter(r1), g2 = fsiv_create_gaussian_filter(r2);        
        g1 = fsiv_extend_image(g1, g2.size(),0); // Extender G[r1] a G[r2]        
        filter = g1-g2; // filtro = G[r2]-G[r1]        
        filter.at<float>(g2.rows/2, g2.cols/2) += 1; // Añadir 1 a la posición central        
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
    
     cv::Mat filter = fsiv_create_sharpening_filter(filter_type, r1, r2); // Crear el filtro

     cv::Size new_size (in.cols+(2*r2), in.rows+(2*r2)); // Tamaño de la imagen extendida
    
     if(only_luma && in.channels() == 3){ // Canal luma
        cv::Mat hsv = in.clone();
        std::vector<cv::Mat> canales;
        cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV); // Convertir a HSV
        cv::split(hsv, canales); // Separar canales
        
        if(circular){ // Expansion circular de la imagen
            cv::Mat circ = fsiv_extend_image(canales[2], new_size, 1); // Extender la imagen circularmente
            cv::filter2D(circ, circ, -1, filter); // Aplicar el filtro
            circ = circ(cv::Rect(r2, r2, in.cols, in.rows)); // Recortar la imagen
            circ.copyTo(canales[2]); // Copiar la imagen recortada al canal luma
        }

        else{ // Expansion de la imagen rellenando con 1
            cv::Mat ext = fsiv_extend_image(canales[2], new_size, 0); // Extender la imagen
            cv::filter2D(ext, ext, -1, filter); // Aplicar el filtro
            ext = ext(cv::Rect(r2, r2, in.cols, in.rows)); // Recortar la imagen
            ext.copyTo(canales[2]); // Copiar la imagen recortada al canal luma
        }
        cv::merge(canales, hsv); // Fusionar los canales
        cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR); // Convertir a BGR
    }

    else{ // Sin canal luma
        cv::Mat img, extended;
        if(circular) extended = fsiv_extend_image(in,new_size,1); // expansion circular
        else extended = fsiv_extend_image(in,new_size,0); // expansion con 1
        cv::filter2D(extended, out, -1, filter); // aplicar el filtro
        out = out(cv::Rect(r2,r2,in.cols,in.rows)); // recortar la imagen
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
 
    std::vector<cv::Point3f> ret_v;
    //Remenber: the first inner point has (1,1) in board coordinates.
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
 
 bool fsiv_find_chessboard_corners(const cv::Mat& img, const cv::Size &board_size,std::vector<cv::Point2f>& corner_points,const char * wname)
{
    
    bool was_found = cv::findChessboardCorners(img,board_size,corner_points); // Buscamos las esquinas del tablero

    if(was_found){ // Esquinas encontradas
        cv::Mat grey = img.clone(); // Clonamos la imagen
        cv::cvtColor(grey, grey, cv::COLOR_BGR2GRAY); // Convertimos la imagen a escala de grises
        // Mejoramos la localizacion de las esquinas
        cv::cornerSubPix(grey, corner_points, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria()); 
    }
    
    if(wname){ // Fichero de salida
        cv::drawChessboardCorners(img, board_size, corner_points, was_found); // Dibujamos las esquinas del tablero
        cv::imshow(wname, img); // Mostramos la imagen
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
    std::vector<cv::Mat> rvecs_, tvecs_;
    // Si los vectores de rotacion y traslacion no son nulos -> Obtenemos el error al calibrar la imagen
    if( (rvecs != nullptr) && (tvecs != nullptr))
        error = cv::calibrateCamera(_3d_points, _2d_points, camera_size, camera_matrix, dist_coeffs, *rvecs, *tvecs);
    
    // Si los vectores de rotacion y traslacion son nulos -> Obtenemos el error al calibrar la imagen
    else
        error = cv::calibrateCamera(_3d_points, _2d_points, camera_size, camera_matrix, dist_coeffs, rvecs_, tvecs_);

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
    
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    // Obtenemos los puntos 3D
    cv::Point3f p0(0, 0, 0), p1(size, 0, 0), p2(0, size, 0), p3(0, 0, -size);
    points3d.push_back(p0);
    points3d.push_back(p1);
    points3d.push_back(p2);
    points3d.push_back(p3);
    cv::projectPoints(points3d, rvec, tvec, camera_matrix, dist_coeffs, points2d); // Realizamos la proyección de los puntos 3D
    // Obtenemos la linea de puntos 2D
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
    
    //Hint: use cv::undistort.
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

    double retard = 1000/fps; // Retardo 
    
    // Obtenemos el tamaño de la imagen
    cv::Size size = cv::Size(input_stream.get(cv::CAP_PROP_FRAME_WIDTH), input_stream.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // Obtenemos la matriz de transformación    
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), camera_matrix, size, CV_32FC1, map1, map2);
    
    input_stream >> frame_in; // Obtenemos la primera imagen
    

    while(!frame_in.empty()){

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
    
    cv::Size tam(2*r+1,2*r+1);
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, tam);
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

    // Imagenes en escala de grises
    cv::Mat previous, cursor;
    cv::cvtColor(prevFrame, previous, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curFrame, cursor, cv::COLOR_BGR2GRAY);
    
    cv::Mat zeros = cv::Mat::zeros(prevFrame.size(), prevFrame.type()); // Generamos la imagen
    
    cv::absdiff(previous, cursor, zeros);  // Diferencia absoluta entre los cursores
    
    difimg = zeros >= thr; // Comparamos con thr
    
    if(r > 0) fsiv_remove_segmentation_noise(difimg, r); // Eliminamos el ruido
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
    
    if(frame.channels()==3) // Frame tiene 3 canales
    {
        masked = cv::Mat::zeros(frame.size(), CV_8UC1); // Creamos la mascara
        cv::cvtColor(mask, masked, cv::COLOR_GRAY2BGR); // Mascara en BGR
    }
    
    else masked = mask; // Frame no tiene 3 canales
    
    outframe = frame & masked; // Aplicamos la mascara
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
    // Hint: convert to input frames to float [0,1].
    // Hint: use cv::accumulate() and cv::accumulateSquare().
    
    cv::Mat frame;
    int key=0, i=0, size=2*gauss_r+1;

    while(was_ok && key!=27 && i < num_frames) // Mientras no se pulse ESC o se alcance el numero de frames
    {
        was_ok=input.read(frame); // Leemos el frame

        if(was_ok) // Frame leido correctamente
        {
            frame.convertTo(frame, CV_32F, 1/255.0); // Frame de tipo flotante [0,1]
            if(gauss_r>0) cv::GaussianBlur(frame, frame, cv::Size(size, size), 0.0); // Filtro Gaussiano

            // Calculamos la media y varianza
            if(mean.empty() || variance.empty())
            {
                mean=frame.clone(); // Clonamos el frame
                variance=frame.mul(frame);
            }

            else
            {
                cv::accumulate(frame, mean);
                cv::accumulateSquare(frame, variance);
            }

            i++; // Incrementamos i
            if(wname) cv::imshow(wname, frame); // Mostramos la imagen
        }
    }

    if(was_ok) // Calculamos la media y la varianza
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
    // Obtenemos la desviacion tipica
    cv::sqrt(variance, sqrt);
    sqrt*=k;
    // Mascara
    cv::Mat masked = diff >= sqrt;
    // Dividimos la mascara en canales
    std::vector<cv::Mat> vector;
    cv::split(masked, vector);
    // Disyuncion entre los elementos del vector de canales
    cv::bitwise_or(vector[0], vector[1], mask);
    cv::bitwise_or(mask, vector[2], mask);
    // Si el radio es positivo, eliminamos el ruido de segmentacion
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

    // Matriz negativa de la mascara
    cv::Mat negative;
    cv::bitwise_not(mask, negative);
    
    // short term updating
    if(short_term_update_period > 0 && frame_count % short_term_update_period == 0){
        // computes a running average of the frames
        cv::accumulateWeighted(frame, mean, alpha, negative);
        cv::accumulateWeighted(frame.mul(frame), variance, alpha, negative);
    }
    
    else if(long_term_update_period > 0 && frame_count % long_term_update_period == 0){
        // computes a running average of the frames
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
    
    // Vector de puntos 3D
    std::vector<cv::Point3f> _3d_points = {
        cv::Point3f(0,0,0), cv::Point3f(size,0,0), cv::Point3f(0,size,0), cv::Point3f(size,size,0), 
        cv::Point3f(size/2,size/2,-size/2)};
    // Vector de puntos 2D
    std::vector<cv::Point2f> _2d_points(_3d_points.size());
    // Proyectamos los puntos
    cv::projectPoints(_3d_points, rvec, tvec, M, dist_coeffs, _2d_points);
    // Unimos los puntos
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
 
 void
fsiv_project_image(const cv::Mat& input, cv::Mat& output,
                   const cv::Size& board_size,
                   const std::vector<cv::Point2f>& _2dpoints)
{
    CV_Assert(!input.empty() && input.type()==CV_8UC3);
    CV_Assert(!output.empty() && output.type()==CV_8UC3);
    CV_Assert(board_size.area()==_2dpoints.size());
    //TODO
    // Creamos la máscara
    cv::Mat mask = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
    // Esquinas del tablero
    std::vector<cv::Point2f> _2dpoints_ = {
               _2dpoints[0], 
               _2dpoints[board_size.width-1], 
               _2dpoints[board_size.height * board_size.width-1], 
               _2dpoints[(board_size.height-1) * board_size.width]
    };
    std::vector<cv::Point> corners = {_2dpoints_[0], _2dpoints_[1], _2dpoints_[2], _2dpoints_[3]};
    // Rellenamos la máscara
    cv::fillConvexPoly(mask, corners, cv::Scalar(255, 255, 255));
    // Transformación de perspectiva
    std::vector<cv::Point2f> _2dpoints_i = {
               cv::Point2f(0, 0), 
               cv::Point2f(input.cols-1, 0), 
               cv::Point2f(input.cols-1, input.rows-1), 
               cv::Point2f(0, input.rows-1)
    };
    cv::Mat transform = cv::getPerspectiveTransform(_2dpoints_i, _2dpoints_);
    // Aplicamos la transformación
    cv::Mat out = cv::Mat::zeros(output.rows, output.cols, CV_8UC1);
    cv::warpPerspective(input, out, transform, output.size());
    // Copiamos la imagen
    out.copyTo(output, mask);
}
 
 # histapathology
 
 /**
 * @brief Create a KNN classifier.
 *
 * @param K is the number of NN neighbours used to classify a sample.
 * @return an instance of the classifier.
 */
 
 cv::Ptr<cv::ml::StatModel>
fsiv_create_knn_classifier(int K)
{
    cv::Ptr<cv::ml::KNearest> knn;

    // TODO: Create an KNN classifier.
    // Hint: Set algorithm type to BRUTE_FORCE.
    // Hint: Set it as a classifier (setIsClassifier)
    // Remenber: Set hyperparameter K.
    knn = cv::ml::KNearest::create();
    knn->setDefaultK(K);
    //

    CV_Assert(knn != nullptr);
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
 
 cv::Ptr<cv::ml::StatModel>
fsiv_create_svm_classifier(int Kernel,
                           double C,
                           double degree,
                           double gamma)
{
    cv::Ptr<cv::ml::SVM> svm;
    // TODO: Create an SVM classifier.
    // Set algorithm type to C_SVC.
    // Set hyperparameters: Kernel, C, Gamma, Degree.
    svm = cv::ml::SVM::create();
    svm->setKernel(Kernel);
    svm->setC(C);
    svm->setDegree(degree);
    svm->setGamma(gamma);
    //
    CV_Assert(svm != nullptr);
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
 
 cv::Ptr<cv::ml::StatModel>
fsiv_create_rtrees_classifier(int V,
                              int T,
                              double E)
{
    cv::Ptr<cv::ml::RTrees> rtrees;
    // TODO: Create an RTrees classifier.
    // Set hyperparameters: Number of features used per node (ActiveVarCount),
    //  max num of trees, and required OOB error.
    // Remenber:: to set T and E parameters use a cv::TerminCriteria object
    //    where T is the max iterations and E is the epsilon.
    rtrees = cv::ml::RTrees::create();
    rtrees->setActiveVarCount(V);
    rtrees->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, T, E));
    //
    CV_Assert(rtrees != nullptr);
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
 
 void fsiv_train_classifier(cv::Ptr<cv::ml::StatModel> &clsf,
                           const cv::Mat &samples, const cv::Mat &labels,
                           int flags)
{
    CV_Assert(clsf != nullptr);

    // TODO: train the classifier.
    // Hint: you can use v::ml::TrainData to set the parameters.
    // Remenber: we are using ROW_SAMPLE ordering in the dataset.
    cv::Ptr<cv::ml::TrainData> data =  cv::ml::TrainData::create(samples,cv::ml::ROW_SAMPLE, labels);
    clsf->train(cv::ml::TrainData::create(samples,cv::ml::ROW_SAMPLE, labels));
    //
    CV_Assert(clsf->isTrained());
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
 
 void fsiv_make_predictions(cv::Ptr<cv::ml::StatModel> &clsf,
                           const cv::Mat &samples, cv::Mat &predictions)
{
    CV_Assert(clsf != nullptr);
    CV_Assert(clsf->isTrained());
    // TODO: do the predictions.
    // Remenber: the classefied used float to save the labels.
    clsf->predict(samples, predictions);
    predictions.convertTo(predictions, CV_32S);
    //
    CV_Assert(predictions.depth() == CV_32S);
    CV_Assert(predictions.rows == samples.rows);
}

/**
 * @brief Load a knn classifier's model from file.
 *
 * @param model_fname is the file name.
 * @return an instance of the classifier.
 * @post ret_v != nullptr
 */
 
 cv::Ptr<cv::ml::StatModel>
fsiv_load_knn_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // TODO: load a KNN classifier.
    // Hint: use the generic interface cv::Algorithm::load< classifier_type >
    clsf = cv::Algorithm::load<cv::ml::KNearest>(model_fname);
    //

    CV_Assert(clsf != nullptr);
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
