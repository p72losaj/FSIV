/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

#include "common_code.hpp"

const char* keys =
    "{help h usage ? |      | print this message.}"
    "{i interactive  |      | Activate interactive mode.}"
    "{l luma         |      | process only \"luma\" if color image.}"
    "{f filter       |0     | filter to use.}"
    "{r1             |1     | r1 for DoG filter.}"
    "{r2             |2     | r2 for DoG filter. (0<r1<r2)}"
    "{c circular     |      | use circular convolution.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}"
    ;

class Parametros{

public:

    Parametros(cv::Mat dentro, int r1, int r2, int filter_type, bool circular, bool luma){

        this->dentro = dentro;
        this->r1 = r1;
        this->r2 = r2;
        this->filter_type = filter_type;
        this->circular = circular;
        this->luma = luma;
        fuera = dentro.clone();
    }
    int r1;
    int r2;
    int filter_type;
    int circular;
    int luma;

    cv::Mat dentro;
    cv::Mat fuera;
};

void trackbar_callback_filter_type(int filter_type, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->filter_type = filter_type;

    cv::Mat mask;

    aux->fuera = fsiv_image_sharpening(aux->dentro, (int)aux->filter_type, (bool)aux->luma, (int)(aux->r1), (int)(aux->r2), (bool)aux->circular);

    cv::imshow("OUTPUT", aux->fuera);
}

void trackbar_callback_luma(int luma, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->luma = luma;

    cv::Mat mask;

    aux->fuera = fsiv_image_sharpening(aux->dentro, (int)aux->filter_type, (bool)aux->luma, (int)(aux->r1), (int)(aux->r2), (bool)aux->circular);

    cv::imshow("OUTPUT", aux->fuera);
}


void trackbar_callback_r1(int r1, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->r1 = r1;

    cv::Mat mask;

    aux->fuera = fsiv_image_sharpening(aux->dentro, (int)aux->filter_type, (bool)aux->luma, (int)(aux->r1), (int)(aux->r2), (bool)aux->circular);

    cv::imshow("OUTPUT", aux->fuera);
}

void trackbar_callback_r2(int r2, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->r2 = r2;

    cv::Mat mask;

    aux->fuera = fsiv_image_sharpening(aux->dentro, (int)aux->filter_type, (bool)aux->luma, (int)(aux->r1), (int)(aux->r2), (bool)aux->circular);

    cv::imshow("OUTPUT", aux->fuera);
}

void trackbar_callback_circular(int circular, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->circular = circular;

    cv::Mat mask;

    aux->fuera = fsiv_image_sharpening(aux->dentro,(int)aux->filter_type, (bool)aux->luma, (int)(aux->r1), (int)(aux->r2), (bool)aux->circular);

    cv::imshow("OUTPUT", aux->fuera);
}

int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Enhance an image using a sharpening filter. (ver 0.0.0)");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }

      cv::String input_name = parser.get<cv::String>(0);
      cv::String output_name = parser.get<cv::String>(1);

      //TODO
      //CLI parameters.
        int r1 = 1, r2 = 2, filter_type = 1;
        bool circular = false, luma = false;

        if(parser.has("i")){
            r1 = 1, r2=2, filter_type = 1, circular = false, luma = false;

            cv::Mat input = cv::imread(input_name, cv::IMREAD_UNCHANGED);
            cv::Mat output = input.clone();

            cv::namedWindow("INPUT");
            cv::namedWindow("OUTPUT");

            Parametros parametros(input, r1, r2, filter_type, circular, luma);
            cv::createTrackbar("r1", "INPUT", &parametros.r1, input.rows, trackbar_callback_r1, &parametros);
            cv::createTrackbar("r2", "INPUT", &parametros.r2, input.rows, trackbar_callback_r2, &parametros);
            cv::createTrackbar("filter_type", "INPUT", &parametros.filter_type, 1, trackbar_callback_filter_type, &parametros);
            cv::createTrackbar("circular", "INPUT", &parametros.circular, 1, trackbar_callback_circular, &parametros);
            cv::createTrackbar("luma", "INPUT", &parametros.luma, 1, trackbar_callback_luma, &parametros);

            cv::imshow("INPUT", input);
            cv::imshow("OUTPUT", parametros.fuera);

            parametros.fuera = fsiv_image_sharpening(input, filter_type, luma, r1, r2, circular);

            int key = cv::waitKey(0) & 0xff;

            if (key != 27)
            {
                if (!cv::imwrite(output_name, parametros.fuera))
                {
                    std::cerr << "Error: could not save the result in file '"<<output_name<<"'."<< std::endl;
                    return EXIT_FAILURE;
                }
            }

            output = parametros.fuera;
            cv::imwrite(output_name, output);
            return retCode;
        }

        else{
            r1 = std::stoi(parser.get<cv::String>("r1"));
            r2 = std::stoi(parser.get<cv::String>("r2"));
            filter_type = std::stoi(parser.get<cv::String>("f"));
            circular = parser.has("c");
            luma = parser.has("l");
        }


      //

      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

      cv::Mat input = cv::imread(input_name);

      if (input.empty())
	  {
		  std::cerr << "Error: could not open the input image '" << input_name << "'." << std::endl;
		  return EXIT_FAILURE;
	  }
      cv::Mat output = input.clone();

      //TODO


      //

      cv::namedWindow("INPUT");
      cv::namedWindow("OUTPUT");


      cv::imshow("INPUT", input);
      cv::imshow("OUTPUT", output);


      int key = cv::waitKey(0) & 0xff;

      //TODO
      //Write the result if it's asked for.

  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
