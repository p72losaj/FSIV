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
    
    // Hint: 

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
    CV_Assert(!in.empty());
    CV_Assert(r>0);
    cv::Mat ret_v;

    ret_v = fsiv_fill_expansion(in, r); // Expansion de la imagen original
    
    // cv::Rect(x,y,w,h). x,y es la esquina superior izquierda, w,h son el ancho y alto del rectangulo

    in(cv::Rect(0,0,in.cols,r)).copyTo(ret_v(cv::Rect(r, ret_v.rows-r, in.cols,r))); // expansion superior
    
    in(cv::Rect(0,in.rows-r,in.cols,r)).copyTo(ret_v(cv::Rect(r, 0, in.cols,r))); // expansion inferior
    
    in(cv::Rect(0,0,r,in.rows)).copyTo(ret_v(cv::Rect(ret_v.cols-r,r,r,in.rows))); // expansion derecha

    in(cv::Rect(in.cols-r,0,r,in.rows)).copyTo(ret_v(cv::Rect(0,r,r,in.rows))); // expansion izquierda
    
    in(cv::Rect(0,0,r,r)).copyTo(ret_v(cv::Rect(ret_v.cols-r,ret_v.rows-r,r,r))); // expansion superior derecha
    
    in(cv::Rect(in.cols-r,in.rows-r,r,r)).copyTo(ret_v(cv::Rect(0,0,r,r))); // expansion inferior izquierda
    
    in(cv::Rect(in.cols-r,0,r,r)).copyTo(ret_v(cv::Rect(0,ret_v.rows-r,r,r))); // expansion inferior derecha

    in(cv::Rect(0,in.rows-r,r,r)).copyTo(ret_v(cv::Rect(ret_v.cols-r,0,r,r))); // expansion superior izquierda

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

