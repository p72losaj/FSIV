/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#include <iostream>
#include <exception>
#include <sstream>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

#include "common_code.hpp"

const cv::String keys =
    "{help h usage ? |      | Print this message}"
    "{r              | x,y,w,h | Use a rectangle (x,y, widht, height)}"
    "{c              | x,y,r | Use a circle (x,y,radius)}"
    "{p              | x1,y1,x2,y2,x3,y3 | Use a closed polygon x1,y1,x2,y2,x3,y3,...}"
    "{i              |     | Interactive mode.}"
    "{@input         | <none> | input image.}"
    "{@output        | <none> | output image.}"
    ;

class Parametros{

    public:
        std::vector<cv::Point> points;
        cv::Point point;
};

void trackbar_callback(int shape, void*)
{
    std::cout << "Opcion marcada: " << shape << " (Pulse intro para continuar o bien seleccione otra opción)" << std::endl;
}

void mouse_callback(int event, int x, int y, int flags, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    if( event == cv::EVENT_LBUTTONDOWN ){
        aux->points.clear();
        aux->point = cv::Point(x, y);
        aux->points.push_back(aux->point);
        std::cout << "X,Y:" << x << "," << y << std::endl;
    }

    if( event == cv::EVENT_LBUTTONUP ){
        aux->point = cv::Point(x, y);
        aux->points.push_back(aux->point);
        std::cout << "X,Y:" << x << "," << y << std::endl;
    }
}

void mouse_callbackpoligono(int event, int x, int y, int flags, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    if( event == cv::EVENT_LBUTTONDOWN ){
        aux->point = cv::Point(x, y);
        aux->points.push_back(aux->point);
        std::cout << "X,Y:" << x << "," << y << std::endl;
    }
}


int
main (int argc, char* const* argv)
{
    int retCode=EXIT_SUCCESS;

    try {
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Do a foreground enhance using a ROI.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }

        cv::String input_n = parser.get<cv::String>("@input");
        cv::String output_n = parser.get<cv::String>("@output");
        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }

        cv::Mat in = cv::imread(input_n, cv::IMREAD_UNCHANGED);
        if (in.empty())
        {
            std::cerr << "Error: could not open input image '" << input_n
                      << "'." << std::endl;
            return EXIT_FAILURE;
        }
        cv::Mat mask = in.clone();
        cv::Mat out = in.clone();

        //TODO

        // Variables a emplear
        char sep;
        int x,y,width,height,radius;
        int shape = 0;
        std::vector<cv::Point> puntos;

        out = convert_rgb_to_gray(in); // Transformamos la imagen de entrada
        out = convert_gray_to_rgb(out); // Obtenemos la imagen de salida
        
        // Modo rectangulo

        if(parser.get<std::string>("r")!="x,y,w,h"){
            // Obtenemos los datos del rectangulo introducidos por el usuario
            std::istringstream buffer(parser.get<std::string>("r"));
            buffer>>x>>sep>>y>>sep>>width>>sep>>height;
            // Caso de error
            if(!buffer){
                std::cerr << "Error:option'-r'incorrect."<<std::endl;
                return EXIT_FAILURE;
            }
            // Generamos la mascara
            mask=generate_rectagle_mask(in.cols,in.rows,x,y,width,height,in.type());
        }

        // Modo circulo

        else if(parser.get<std::string>("c")!="x,y,r"){
            // Obtenemos los datos del circulo       
            std::istringstream buffer(parser.get<std::string>("c"));
            buffer>>x>>sep>>y>>sep>>radius;
            // Caso de error
            if(!buffer){
                std::cerr << "Error:option'-c'incorrect."<<std::endl;
                return EXIT_FAILURE;
            }
            mask=generate_circle_mask(in.cols,in.rows,x,y,radius,in.type()); // genera la mascara del rectangulo
        }

        // Modo poligono cerrado

        else if(parser.get<std::string>("p")!="x1,y1,x2,y2,x3,y3"){
            std::istringstream buffer(parser.get<std::string>("p"));
            // Caso de error
            if(!buffer){
                std::cerr << "Error:option'-p'incorrect."<<std::endl;
                return EXIT_FAILURE;
            }

            do{
                buffer >> x >> sep >> y >> sep;
                cv::Point punto(x,y); // Creamos el punto
                puntos.push_back(punto); // Guardamos el punto en el vector de puntos del poligono
            } while(buffer);

            if(puntos.size() > 2){
                mask=generate_polygon_mask(in.cols,in.rows,puntos,in.type());
            }

            else{
                std::cerr << "No es poligono" << std::endl;
            }

        }

        // Modo interactivo

        else if (parser.has("i"))
                {
                    Parametros parametros;
                    cv::namedWindow("EDITOR", cv::WINDOW_GUI_EXPANDED);
                    cv::createTrackbar("funcionalidad", "EDITOR", &shape, 2, trackbar_callback);
                    std::cout << "Opciones: " << std::endl;
                    std::cout << "0. Rectangulo. " << std::endl;
                    std::cout << "1. Circulo. " << std::endl;
                    std::cout << "2. Polígono. " << std::endl;
                    std::cout << "Opcion marcada: " << shape << " (Pulse intro para continuar o bien seleccione otra opción)" << std::endl;
                    cv::imshow("EDITOR", in);
                    int k = cv::waitKey(0)&0xff;
                    if (k!=27){
                        if(shape == 0){
                            std::cout << "Opcion elegida: " << shape << " (Pulse y arrastre para generar el rectangulo y una vez haya terminado pulse la tecla intro.)" << std::endl;
                            cv::setMouseCallback("EDITOR", mouse_callback, &parametros);
                            int k = cv::waitKey(0)&0xff;
                            if (k!=27){
                                width = parametros.points[parametros.points.size()-1].x - parametros.points[parametros.points.size()-2].x;
                                height = parametros.points[parametros.points.size()-1].y - parametros.points[parametros.points.size()-2].y;
                                if(width < 0 && height < 0){
                                    mask = generate_rectagle_mask(in.cols, in.rows, parametros.points[parametros.points.size()-1].x, parametros.points[parametros.points.size()-1].y, -width, -height, in.type());
                                }
                                else if(width < 0 && height >= 0 ){
                                    mask = generate_rectagle_mask(in.cols, in.rows, parametros.points[parametros.points.size()-1].x, parametros.points[parametros.points.size()-2].y, -width, height, in.type());
                                }
                                else if(width >= 0 && height < 0){
                                    mask = generate_rectagle_mask(in.cols, in.rows, parametros.points[parametros.points.size()-2].x, parametros.points[parametros.points.size()-1].y, width, -height, in.type());
                                }
                                else{
                                    mask = generate_rectagle_mask(in.cols, in.rows, parametros.points[parametros.points.size()-2].x, parametros.points[parametros.points.size()-2].y, width, height, in.type());
                                }
                            }
                        }

                        else if (shape == 1){
                            std::cout << "Opcion elegida: " << shape << " (Pulse y arrastre para generar el circulo y una vez haya terminado pulse la tecla intro.)" << std::endl;
                            cv::setMouseCallback("EDITOR", mouse_callback, &parametros);
                            int k = cv::waitKey(0)&0xff;
                            if (k!=27){
                                width = parametros.points[parametros.points.size()-1].x - parametros.points[parametros.points.size()-2].x;
                                height = parametros.points[parametros.points.size()-1].y - parametros.points[parametros.points.size()-2].y;
                                radius = sqrt(((width*width)+(height*height)));
                                mask = generate_circle_mask(in.cols, in.rows, parametros.points[parametros.points.size()-2].x, parametros.points[parametros.points.size()-2].y, radius, in.type());
                            }
                        }

                        else if (shape == 2){
                            std::cout << "Opcion elegida: " << shape << " (Pulse para crear los puntos del poligono y una vez haya terminado pulse la tecla intro.)" << std::endl;
                            cv::setMouseCallback("EDITOR", mouse_callbackpoligono, &parametros);
                            int k = cv::waitKey(0)&0xff;
                            if (k!=27){
                                mask = generate_polygon_mask(in.cols, in.rows, parametros.points, in.type());
                            }
                        }
                    }

                    cv::destroyWindow("EDITOR");
                }



        // Combinamos las imagenes
        out = combine_images(in,out,mask);


        //

        cv::namedWindow("INPUT", cv::WINDOW_GUI_EXPANDED);
        cv::imshow("INPUT", in);
        // cv::namedWindow("MASK",  cv::WINDOW_GUI_EXPANDED);
        // cv::imshow("MASK", mask);
        cv::namedWindow("OUTPUT",  cv::WINDOW_GUI_EXPANDED);
        cv::imshow("OUTPUT", out);

        int k = cv::waitKey(0)&0xff;
        if (k!=27)
            cv::imwrite(output_n, out);
    }
    catch (std::exception& e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
