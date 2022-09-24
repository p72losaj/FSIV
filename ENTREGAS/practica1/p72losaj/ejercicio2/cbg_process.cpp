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
    "{i interactive  |      | Activate interactive mode.}"
    "{l luma         |      | process only \"luma\" if color image.}"
    "{c contrast     |1.0   | contrast parameter.}"
    "{b bright       |0.0   | bright parameter.}"
    "{g gamma        |1.0   | gamma parameter.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}"
    ;

class Parametros{

public:

    Parametros(cv::Mat dentro, int contraste, int brillo, int gama, int luma){

        this->dentro = dentro;
        this->contraste = contraste;
        this->brillo = brillo;
        this->gama = gama;
        this->luma = luma;
        fuera = dentro.clone();
    }

    int contraste;
    int brillo;
    int gama;
    int luma;

    cv::Mat dentro;
    cv::Mat fuera;
};

void trackbar_callback_contraste(int contraste, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->contraste = contraste;

    std::cout << "Contraste actual: " << contraste << std::endl;

    std::cout << "Contraste: " << (double)(aux->contraste)/100 << " Brillo: " << (double)(aux->brillo-100)/100 << " Gama: " << (double)aux->gama/100 << " Luma: " <<
                 aux->luma << std::endl;

    cbg_process(aux->dentro, aux->fuera, (double)(aux->contraste)/100, (double)(aux->brillo-100)/100, (double)aux->gama/100, aux->luma);
    cv::imshow("PROCESADA", aux->fuera);
}

void trackbar_callback_brillo(int brillo, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->brillo = brillo;

    std::cout << "Brillo actual: " << brillo << std::endl;

    std::cout << "Contraste: " << (double)(aux->contraste)/100 << " Brillo: " << (double)(aux->brillo-100)/100 << " Gama: " << (double)aux->gama/100 << " Luma: " <<
                 aux->luma << std::endl;

    cbg_process(aux->dentro, aux->fuera, (double)(aux->contraste)/100, (double)(aux->brillo-100)/100, (double)aux->gama/100, aux->luma);
    cv::imshow("PROCESADA", aux->fuera);
}

void trackbar_callback_gamma(int gama, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->gama = gama;

    std::cout << "Gamma actual: " << gama << std::endl;

    std::cout << "Contraste: " << (double)(aux->contraste)/100 << " Brillo: " << (double)(aux->brillo-100)/100 << " Gama: " << (double)aux->gama/100 << " Luma: " <<
                 aux->luma << std::endl;

    cbg_process(aux->dentro, aux->fuera, (double)(aux->contraste)/100, (double)(aux->brillo-100)/100, (double)aux->gama/100, aux->luma);
    cv::imshow("PROCESADA", aux->fuera);
}

void trackbar_callback_luma(int luma, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->luma = luma;

    std::cout << "Luma actual: " << luma << std::endl;

    std::cout << "Contraste: " << (double)aux->contraste/100 << " Brillo: " << (double)(aux->brillo-100)/100 << " Gama: " << (double)aux->gama/100 << " Luma: " <<
                 aux->luma << std::endl;

    cbg_process(aux->dentro, aux->fuera, (double)aux->contraste/100, (double)(aux->brillo-100)/100, (double)aux->gama/100, aux->luma);
    cv::imshow("PROCESADA", aux->fuera);
}



int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Ajust the contrast/bright/gamma parameters of an image. (ver 0.0.0)");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }

      cv::String input_name = parser.get<cv::String>(0);
      cv::String output_name = parser.get<cv::String>(1);

      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

      cv::Mat input;
      cv::Mat output;
      cv::namedWindow("ORIGINAL");
      cv::namedWindow("PROCESADA");


      //TODO

      if (parser.has("i")){

          cv::String input_n = parser.get<cv::String>("@input");
          cv::String output_n = parser.get<cv::String>("@output");

          input = cv::imread(input_n, cv::IMREAD_UNCHANGED);

          Parametros parametros(input.clone(), 100, 100, 100, 0);

          cv::createTrackbar("contraste", "ORIGINAL", &parametros.contraste, 200, trackbar_callback_contraste, &parametros);
          cv::createTrackbar("brillo", "ORIGINAL", &parametros.brillo, 200, trackbar_callback_brillo, &parametros);
          cv::createTrackbar("gamma", "ORIGINAL", &parametros.gama, 200, trackbar_callback_gamma, &parametros);
          cv::createTrackbar("luma", "ORIGINAL", &parametros.luma, 1, trackbar_callback_luma, &parametros);

          cv::imshow("ORIGINAL", input);

          cbg_process(parametros.dentro, parametros.fuera, (double)parametros.contraste/100, (double)(parametros.brillo-100)/100, (double)parametros.gama/100, parametros.luma);
          cv::imshow("PROCESADA", parametros.fuera);

          int key = cv::waitKey(0) & 0xff;

          if (key != 27)
          {
              if (!cv::imwrite(output_n, parametros.fuera))
              {
                  std::cerr << "Error: could not save the result in file '"<<output_n<<"'."<< std::endl;
                    return EXIT_FAILURE;
                }
          }
      }

      else{

          cv::String input_n = parser.get<cv::String>("@input");
          cv::String output_n = parser.get<cv::String>("@output");

          input = cv::imread(input_n, cv::IMREAD_UNCHANGED);

          cv::imshow("ORIGINAL", input);

          double contraste = std::stod(parser.get<cv::String>("c"));
          double brillo = std::stod(parser.get<cv::String>("b"));
          double gama = std::stod(parser.get<cv::String>("g"));
          int luma = parser.has("l");

          std::cout << "Contraste: " << contraste << " Brillo: " << brillo << " Gama: " << gama << " Luma: " << luma << std::endl;

          cbg_process(input, output, contraste, brillo, gama, luma);
          cv::imshow("PROCESADA", output);

          int key = cv::waitKey(0) & 0xff;

          if (key != 27)
          {
              if (!cv::imwrite(output_n, output))
              {
                  std::cerr << "Error: could not save the result in file '"<<output_n<<"'."<< std::endl;
                    return EXIT_FAILURE;
                }
          }
      }



  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}


