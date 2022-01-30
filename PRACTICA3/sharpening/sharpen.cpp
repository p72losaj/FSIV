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

const cv::String keys =
    "{help h usage ? |      | print this message.}"
    "{i interactive  |      | Activate interactive mode.}"
    "{l luma         |      | process only \"luma\" if color image.}"
    "{f filter       |0     | filter to use.}"
    "{g gain         |0.0   | filter gain.}"
    "{r1             |1     | r1 for DoG filter.}"
    "{r2             |2     | r2 for DoG filter. (0<r1<r2)}"
    "{c circular     |      | use circular convolution.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}"
    ;


int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Ajust the contrast/bright/gamma parameters of a image. (ver 0.0.0)");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }

      cv::String input_name = parser.get<cv::String>(0);
      cv::String output_name = parser.get<cv::String>(1);

      //TODO
      //CLI parameters.
        // circular convolution
        bool circular = false;
        // Filter
        int filter = 0;
        // Ganancia
        double g = 0.0;
        // r1
        int r1=1;
        // r2
        int r2=2;
        // Canal luma
        int l=0;
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
      // Comprobamos si se desea aplicar una convolucion circular
      if(parser.has("c")) {circular = true;}
      // Comprobamos proceso del canal luma
      if(parser.has("l")){l=1;}
      // Comprobamos si hay un valor r1
      if(parser.get<std::string>("r1") != "x"){
          std::istringstream buffer(parser.get<std::string>("r1"));
          buffer >> r1;
          if(!buffer){
              std::cerr << "Error: option '-r1' incorrect" << std::endl;
              return EXIT_FAILURE;
          }
      }
      // Comprobamos si hay un valor r2
      if(parser.get<std::string>("r2") != "x"){
          std::istringstream buffer(parser.get<std::string>("r2"));
          buffer >> r2;
          if(!buffer){
              std::cerr << "Error: option '-r2' incorrect" << std::endl;
              return EXIT_FAILURE;
          }
      }
      // Comprobamos si se desea aplicar una ganancia
      if(parser.get<std::string>("g") != "x"){
          std::istringstream buffer(parser.get<std::string>("g"));
          buffer >> g;
          if(!buffer){
              std::cerr << "Error: option 'g' incorrect" << std::endl;
              return EXIT_FAILURE;
          }
      }
      // Filtro a aplicar
      if(parser.get<std::string>("f")!="x")
      {
          std::istringstream buffer(parser.get<std::string>("f"));
          buffer >> filter;
          if(!buffer)
          {
              std::cerr << "Error:option'-f'incorrect."<<std::endl;
              return EXIT_FAILURE;
          }

          if(filter != 0 && filter != 1 && filter != 2){
              std::cerr << "Filtro no valido" << std::endl;
              filter = 0;
          }
      }


        // Transformamos la imagen a HSV

        cv::Mat in = input.clone();

        cv::cvtColor(in,in, cv::COLOR_BGR2HSV);

        // Creamos un vector de canales

        std::vector<cv::Mat> canales;

        //Guardamos cada canal como una matriz diferente. Lo enlazamos con la imagen1
        cv::split(in, canales);
        // Aplicamos usm_enhance en la imagen de mascara
        canales[2] = fsiv_image_sharpening(canales[2], g,filter,l,r1,r2,circular);
        cv::merge(canales,output);
        cv::cvtColor(output,output,cv::COLOR_HSV2BGR);

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
