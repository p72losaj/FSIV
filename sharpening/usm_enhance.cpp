/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

#include "common_code.hpp"

const cv::String keys =
    "{help h usage ? |      | print this message.}"
    "{i interactive  |      | Interactive mode.}"
    "{r radius       |1     | Window's radius. Default 1.}"
    "{g gain         |1.0   | Enhance's gain. Default 1.0}"
    "{c circular     |      | Use circular convolution.}"
    "{f filter       |0     | Filter type: 0->Box, 1->Gaussian. Default 0.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}"
    ;

class Parametros{

public:

    Parametros(cv::Mat dentro, double g, int r, int filter_type, bool circular){

        this->dentro = dentro;
        this->g = g;
        this->r = r;
        this->filter_type = filter_type;
        this->circular = circular;
        fuera = dentro.clone();
    }

    int g;
    int r;
    int filter_type;
    int circular;

    cv::Mat dentro;
    cv::Mat fuera;
    cv::Mat mask;
};

void trackbar_callback_g(int g, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->g = g;

    cv::Mat mask;

    if(aux->dentro.channels()==3){

        cv::cvtColor(aux->dentro, aux->dentro, cv::COLOR_BGR2HSV);
        std::vector <cv::Mat> channels;
        cv::split(aux->dentro, channels);
        channels[2] = fsiv_usm_enhance(channels[2], (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
        cv::merge(channels, aux->fuera);
        cv::cvtColor(aux->fuera, aux->fuera, cv::COLOR_HSV2BGR);
    }

    else{

        aux->fuera = fsiv_usm_enhance(aux->dentro, (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
    }

    //fsiv_usm_enhance(aux->dentro, (int)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);

    aux->mask = mask;
    cv::imshow("OUTPUT", aux->fuera);
    cv::imshow("UNSHARP MASK", aux->mask);
}

void trackbar_callback_filter_type(int filter_type, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->filter_type = filter_type;

    cv::Mat mask;

    if(aux->dentro.channels()==3){

        cv::cvtColor(aux->dentro, aux->dentro, cv::COLOR_BGR2HSV);
        std::vector <cv::Mat> channels;
        cv::split(aux->dentro, channels);
        channels[2] = fsiv_usm_enhance(channels[2], (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
        cv::merge(channels, aux->fuera);
        cv::cvtColor(aux->fuera, aux->fuera, cv::COLOR_HSV2BGR);
    }

    else{

        aux->fuera = fsiv_usm_enhance(aux->dentro, (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
    }

    aux->mask = mask;
    cv::imshow("OUTPUT", aux->fuera);
    cv::imshow("UNSHARP MASK", aux->mask);
}

void trackbar_callback_circular(int circular, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->circular = circular;

    cv::Mat mask;

    if(aux->dentro.channels()==3){

        cv::cvtColor(aux->dentro, aux->dentro, cv::COLOR_BGR2HSV);
        std::vector <cv::Mat> channels;
        cv::split(aux->dentro, channels);
        channels[2] = fsiv_usm_enhance(channels[2], (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
        cv::merge(channels, aux->fuera);
        cv::cvtColor(aux->fuera, aux->fuera, cv::COLOR_HSV2BGR);
    }

    else{

        aux->fuera = fsiv_usm_enhance(aux->dentro, (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
    }

    aux->mask = mask;
    cv::imshow("OUTPUT", aux->fuera);
    cv::imshow("UNSHARP MASK", aux->mask);
}

void trackbar_callback_r(int r, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->r = r;

    cv::Mat mask;

    if(aux->dentro.channels()==3){

        cv::cvtColor(aux->dentro, aux->dentro, cv::COLOR_BGR2HSV);
        std::vector <cv::Mat> channels;
        cv::split(aux->dentro, channels);
        channels[2] = fsiv_usm_enhance(channels[2], (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
        cv::merge(channels, aux->fuera);
        cv::cvtColor(aux->fuera, aux->fuera, cv::COLOR_HSV2BGR);
    }

    else{

        aux->fuera = fsiv_usm_enhance(aux->dentro, (double)(aux->g)/100, (int)(aux->r), (int)aux->filter_type, (bool)aux->circular, &mask);
    }

    aux->mask = mask;
    cv::imshow("OUTPUT", aux->fuera);
    cv::imshow("UNSHARP MASK", aux->mask);
}

int
main (int argc, char* const* argv)
{
    int retCode=EXIT_SUCCESS;

    try {
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Apply an unsharp mask enhance to an image.");
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

        //TODO
        double g = 1.0;
        int r = 1;
        int filter_type=1;
        bool circular =false;

        cv::Mat input = cv::imread(input_n, cv::IMREAD_UNCHANGED);

        if(parser.has("i")){
            cv::namedWindow("INPUT");
            cv::namedWindow("OUTPUT");
            cv::namedWindow("UNSHARP MASK");
            g = 100;
            r = 1;
            filter_type=1;
            circular = false;
            cv::Mat input = cv::imread(input_n, cv::IMREAD_UNCHANGED);
            input.convertTo(input, CV_32FC1, 1/255.0);
            cv::Mat mask = input.clone();
            cv::Mat output = input.clone();
            Parametros parametros(input, g, r, filter_type, circular);
            cv::createTrackbar("g", "INPUT", &parametros.g, 500, trackbar_callback_g, &parametros);
            cv::createTrackbar("r", "INPUT", &parametros.r, input.rows, trackbar_callback_r, &parametros);
            cv::createTrackbar("filter_type", "INPUT", &parametros.filter_type, 1, trackbar_callback_filter_type, &parametros);
            cv::createTrackbar("circular", "INPUT", &parametros.circular, 1, trackbar_callback_circular, &parametros);
            cv::imshow("INPUT", input);

            if(parametros.dentro.channels()==3){
                cv::cvtColor(parametros.dentro, parametros.dentro, cv::COLOR_BGR2HSV);
                std::vector <cv::Mat> canales;
                cv::split(parametros.dentro, canales);
                canales[2] = fsiv_usm_enhance(canales[2], g, r, filter_type, circular, &mask);
                cv::merge(canales, parametros.fuera);
                cv::cvtColor(parametros.fuera, parametros.fuera, cv::COLOR_HSV2BGR);
            }

            else{
                parametros.fuera = fsiv_usm_enhance(parametros.dentro, g, r, filter_type, circular, &mask);
            }

            cv::imshow("OUTPUT", parametros.fuera);
            cv::imshow("UNSHARP MASK", mask);

            int key = cv::waitKey(0) & 0xff;

            if (key != 27)
            {
                if (!cv::imwrite(output_n, parametros.fuera))
                {
                    std::cerr << "Error: could not save the result in file '"<<output_n<<"'."<< std::endl;
                    return EXIT_FAILURE;
                }
            }

            output = parametros.fuera;
            cv::imwrite(output_n, output);
            return retCode;
        }

        else{
            g = std::stod(parser.get<cv::String>("g"));
            r = std::stoi(parser.get<cv::String>("r"));
            filter_type = std::stoi(parser.get<cv::String>("f"));
            circular = parser.has("c");
        }
        //

        cv::Mat in = cv::imread(input_n, cv::IMREAD_UNCHANGED);
        cv::Mat out = in.clone();
        cv::Mat mask = in.clone();

        if (in.empty())
        {
            std::cerr << "Error: could not open input image '" << input_n
                      << "'." << std::endl;
            return EXIT_FAILURE;
        }

        cv::namedWindow("INPUT");
        cv::namedWindow("OUTPUT");
        cv::namedWindow("UNSHARP MASK");

        in.convertTo(in, CV_32FC1, 1/255.0);

        if(in.channels()==3){
            // Transformamos la imagen BGR en HSV
            cv::cvtColor(in, in, cv::COLOR_BGR2HSV);
            std::vector <cv::Mat> channels;
            cv::split(in, channels);
            // Aplicamos usm_enhance al canal V
            channels[2] = fsiv_usm_enhance(channels[2], g, r, filter_type, circular, &mask);
            cv::merge(channels, out);
            // Transformamos la imagen HSV en BGR
            cv::cvtColor(out, out, cv::COLOR_HSV2BGR);
        }

        else{

            out = fsiv_usm_enhance(in, g, r, filter_type, circular, &mask);
        }



        cv::imshow ("INPUT", in);
        cv::imshow ("OUTPUT", out);
        cv::imshow ("UNSHARP MASK", mask);

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
