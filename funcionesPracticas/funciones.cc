#include <iostream>
#include <exception>
#include <valarray>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

#include "funciones.hpp"

void mostrarMenu(){
	std::cout << "#######################" << std::endl;
	std::cout << "0. Finalizar el programa" << std::endl;
	std::cout << "1. compute_stats" << std::endl;
    std::cout << "2. esqueleto" << std::endl;
    std::cout << "3. show_image" << std::endl;
    std::cout << "4. show_video" << std::endl;
	std::cout << "Introduce una opcion: ";
}

void on_mouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        static_cast<int*>(userdata)[0] = x;
        static_cast<int*>(userdata)[1] = y;
    }
}

void show_video(const std::string video_name, const int opcion_video, const int camera_idx, const int wait){
    cv::VideoCapture vid;
    cv::namedWindow("VIDEO"); // Ventana del video  
    cv::Mat frame;
    if (opcion_video == 1) vid.open(video_name); 
    else vid.open(camera_idx);
    if (!vid.isOpened())
    {
        std::cerr << "Error: no he podido abrir el la fuente de vídeo." << std::endl;
        return;
    }
    vid >> frame; // Captura el primer frame.
    if (frame.empty()) // Si el frame esta vacio, puede ser un error hardware o fin del video.
    {
        std::cerr << "Error: could not capture any frame from source." << std::endl;
        return;
    }
    std::cout << "Input size (WxH): " << frame.cols << 'x' << frame.rows << std::endl;
    std::cout << "Frame rate (fps): " << vid.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "Num of frames   : " << vid.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    // Coordenadas del pixel a muestrear. Inicialmente muestrearemos el pixel central.
    int coords[2] = {frame.cols/2, frame.rows/2};
    cv::setMouseCallback ("VIDEO", on_mouse, coords);
    std::cerr << "Pulsa una tecla para continuar (ESC para salir)." << std::endl;
    int key = cv::waitKey(0) & 0xff;
    while (!frame.empty() && key!=27)
    {
        cv::imshow("VIDEO", frame);//muestro el frame.
        const cv::Vec3b v = frame.at<cv::Vec3b>(coords[1], coords[0]); //mostramos los valores RGB del pixel muestreado.
        std::cout << "RGB point (" << coords[0] << ',' << coords[1] << "): "
                   << static_cast<int>(v[0]) << ", "
                   << static_cast<int>(v[1]) << ", "
                   << static_cast<int>(v[2]) << std::endl;
        // Espero un tiempo fijado. Si el usuario pulsa una tecla obtengo el codigo ascci. Si pasa el tiempo, retorna -1.
        key = cv::waitKey(wait) & 0xff;
        vid >> frame;// capturo el siguiente frame.
    }
}

void show_image(const cv::Mat& img){
    cv::namedWindow("VENTANA",cv::WINDOW_GUI_EXPANDED); // Creamos la ventana de la imagen
	cv::imshow("VENTANA",img); // Mostramos la imagen en la ventana
	std::cout << "Pulsa ESC para salir." << std::endl;
    while ((cv::waitKey(0) & 0xff) != 27); //Hasta que no se pulse la tecla ESC no salimos.
}

void esqueleto(const cv::Mat& img){
    cv::Mat copia = img.clone(); // Realizamos una copia de la imagen
    std::vector<cv::Mat> canales; // Vector de canales de la imagen
    cv::split(copia,canales); // Obtenemos los canales de la imagen
    double min=0.0; // valor minimo del canal
    double max = 0.0; // valor maximo del canal
    for(size_t i=0; i < canales.size(); i++){
        cv::minMaxIdx(canales[i],&min,&max);
        std::cout << "Canal " << i << " -> Min: " << min << "; Max: " << max << std::endl;
    }
}

void comp_stats(const cv::Mat& img){
    float media = 0.0;
    float dev = 0.0;
    cv::Mat copia = img.clone();
    std::cout << "Ancho de la imagen: " << copia.cols << std::endl;
    std::cout << "Alto  de la imagen: " << copia.rows << std::endl;
    std::cout << "Número de canales de la imagen: " << copia.channels() << std::endl;
    std::cout << "Profundidad de bit de la imagen: ";
    switch (copia.depth())
    {
        case CV_8S:
          std::cout << " entero 8 bits con signo." << std::endl;
          break;
        case CV_8U:
          std::cout << " entero 8 bits sin signo." << std::endl;
          break;
        case CV_16S:
          std::cout << " entero 16 bits con signo." << std::endl;
          break;
        case CV_16U:
          std::cout << " entero 16 bits sin signo." << std::endl;
          break;
        case CV_32S:
          std::cout << " entero 32 bits con signo." << std::endl;
          break;
        case CV_32F:
          std::cout << " flotante 32 bits." << std::endl;
          break;
        case CV_64F:
          std::cout << " flotante 64 bits." << std::endl;
          break;
        default:
          std::cout << " otros tipos." << std::endl;
          break;
    }
    std::vector<cv::Mat> canales; // canales de la imagen
    cv::split(copia,canales);
    std::cout << "Utilizando la opcion 1" << std::endl;
    for(size_t i=0; i<canales.size();i++){
        compute_stats1(canales[i],media,dev);
        std::cout << "Canal " << i <<  " Media: "<<media<<"; Desviacion tipica: "<<dev<<std::endl;
    }
    std::cout << "################" << std::endl;
    std::cout << "Utilizando la opcion 2" << std::endl;
    for(size_t i=0; i < canales.size(); i++){
        compute_stats2(canales[i], media, dev);
        std::cout << "Canal " << i <<  " Media: "<<media<<"; Desviacion tipica: "<<dev<<std::endl; 
    }
    std::cout << "################" << std::endl;  
    for(size_t i=0; i < canales.size(); i++) 
        canales[i].convertTo(copia,CV_32FC1);// Transformamos los canales a flotante
    std::cout << "Utilizando la opcion 3" << std::endl; 
    for(size_t i=0; i < canales.size(); i++){
        compute_stats3(copia, media, dev);
        std::cout << "Canal " << i << " Media: " << media << "; Desviacion tipica: " << dev << std::endl;
    }
    std::cout << "################" << std::endl;  
    std::cout << "Utilizando la opcion 4" << std::endl;
    for(size_t i=0; i < canales.size(); i++){
        compute_stats4(copia, media, dev);
        std::cout << "Canal " << i << " Media: " << media << "; Desviacion tipica: " << dev << std::endl;
    }
}

void 
compute_stats1(const cv::Mat& img, float& media, float& dev)
{
    //Comprobacion de precondiciones.
    CV_Assert( !img.empty() );
    CV_Assert( img.type() == CV_8UC1 );
    media = 0.0;
    dev = 0.0;
    //Para cada fila 0 ... img.rows-1
    for (int row=0; row<img.rows; ++row)
        //Para cada columna 0 ... img.cols-1
        for (int col=0; col<img.cols; ++col)
        {
            //Acceder a un pixel con el metodo at<Tipo de pixel>(fila, columna).
            //Aqui cada pixel es un byte (uchar).

            const float v = img.at<uchar>(row, col);

            //También podríamos tener otros tipos:
            //   Tres bytes por pixel (CV_8UC3) ->  at<cv::Vec3b>
            //   Un solo float por pixel (CV_32FC1)  ->  at<float>
            //   tres floats por pixel (CV_32FC3)   -> at<cv::Vec3f>
            //   ... más combinaciones.
            media += v;
            dev += v*v;
        }
    const float count = img.rows*img.cols;
    media /= count;
    dev /= count;
    dev = cv::sqrt(dev - media*media);
}

void
compute_stats2(const cv::Mat& img, float& media, float& dev)
{
    //Comprobacion de precondiciones.
    CV_Assert( !img.empty() );
    CV_Assert( img.type() == CV_8UC1 );
    media = 0.0;
    dev = 0.0;
    const auto end = img.end<uchar>();
    for (auto p = img.begin<uchar>(); p != end; ++p )
    {
        //También podríamos tener otros tipos:
        //   Tres bytes por pixel (CV_8UC3) ->  begin<cv::Vec3b>()
        //   Un solo float por pixel (CV_32FC1)  ->  begin<float>()
        //   tres floats por pixel (CV_32FC3)   -> begin<cv::Vec3f>()
        //   ... mas combinaciones.
        const float v = *p;
        media += v;
        dev += v*v;
    }
    const float count = img.rows*img.cols;
    media /= count;
    dev /= count;
    dev = cv::sqrt(dev - media*media);
}

void
compute_stats3(const cv::Mat& img, float& media, float& dev)
{
    //Comprobacion de precondiciones.
    CV_Assert( !img.empty() );
    CV_Assert( img.depth() == CV_32FC1 );
    media = static_cast<float>(cv::sum(img)[0]);
    dev = static_cast<float>(cv::sum(img.mul(img))[0]);
    const float count = img.rows*img.cols;
    media /= count;
    dev /= count;
    dev = cv::sqrt(dev - media*media);
}

void
compute_stats4(const cv::Mat& img, float& media, float& dev)
{
    //Comprobacion de precondiciones.
    CV_Assert( !img.empty() );
    CV_Assert( img.type() == CV_32FC1 );
    cv::Scalar mean;
    cv::Scalar stdev;
    cv::meanStdDev(img, mean, stdev);
    media = static_cast<float>(mean[0]);
    dev = static_cast<float>(stdev[0]);
}