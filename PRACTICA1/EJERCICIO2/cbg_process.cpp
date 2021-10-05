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
      //cv::namedWindow("ORIGINAL");
      //cv::namedWindow("PROCESADA");

      //TODO

      // Abrimos la imagen de entrada

      input = cv::imread(input_name, cv::IMREAD_UNCHANGED);

      if (input.empty())
      {
          std::cerr << "Error: could not open input image '" << input_name
                    << "'." << std::endl;
          return EXIT_FAILURE;
      }

      // Obtenemos la mascara de la imagen de entrada

      output = input.clone();

      cv::namedWindow("ORIGINAL", cv::WINDOW_GUI_EXPANDED);

      cv::imshow("ORIGINAL", input);

      // Procesado de contraste

      float contraste = 1.0; // Valor del contraste por defecto

      float brillo = 0.0; // Valor del brillo por defecto

      float gamma = 1.0; // Valor por defecto de gamma

      bool luma = false; // Por defecto, no se procesa el canal luma

      // Comando -c
      if(parser.get<std::string>("c")!="x"){ // El usuario ha puesto -c en la lista de comandos
        float x;
        std::istringstream buffer(parser.get<std::string>("c"));
        buffer >> x; // Obtenemos el valor flotante del buffer
        // Comprobamos si el buffer esta vacio
        if(!buffer){
            std::cerr << "Error: Option -c incorrecta" << std::endl;
            return EXIT_FAILURE;
        }
        // El valor pasado en el -c debe ser flotante en el rango [0,2.0]

        if(x < 0.0 || x > 2.0){
            std::cerr << "El valor pasado en -c debe ser flotante en el rango [0,2.0]" << std::endl;
            std::cerr << "Valor por defecto del contraste utilizado: " <<contraste << std::endl;
        }

        else{
            contraste = x;
        }


      }

      // Comando -b

      if(parser.get<std::string>("b")!="x"){ // El usuario ha puesto -c en la lista de comandos
        float x;
        std::istringstream buffer(parser.get<std::string>("b"));
        buffer >> x; // Obtenemos el valor flotante del buffer
        // Comprobamos si el buffer esta vacio
        if(!buffer){
            std::cerr << "Error: Option -b incorrecta" << std::endl;
            return EXIT_FAILURE;
        }
        // El valor pasado en el -b debe ser flotante en el rango [-1.0,1.0]

        if(x < -1.0 || x > 1.0){
            std::cerr << "El valor pasado en -b debe ser flotante en el rango [-1.0,1.0]" << std::endl;
            std::cerr << "Valor por defecto del brillo utilizado: " <<brillo << std::endl;
        }

        else{
            brillo = x;
        }


      }

      // comando gamma

      if(parser.get<std::string>("g")!="x"){ // El usuario ha puesto -g en la lista de comandos
        float x;
        std::istringstream buffer(parser.get<std::string>("g"));
        buffer >> x; // Obtenemos el valor flotante del buffer
        // Comprobamos si el buffer esta vacio
        if(!buffer){
            std::cerr << "Error: Option -g incorrecta" << std::endl;
            return EXIT_FAILURE;
        }
        // El valor pasado en el -g debe ser flotante en el rango [0.0,2.0]

        if(x < 0.0 || x > 2.0){
            std::cerr << "El valor pasado en -g debe ser flotante en el rango [0,2.0]" << std::endl;
            std::cerr << "Valor por defecto del contraste utilizado: " <<gamma << std::endl;
        }

        else{
            gamma = x;
        }


      }




      // Realizamos el cambio

      output = cbg_process (input, output,contraste,brillo, gamma,luma);

      // Mostramos la imagen

      cv::namedWindow("PROCESADA", cv::WINDOW_GUI_EXPANDED);

      cv::imshow("PROCESADA", output);

      int key = cv::waitKey(0) & 0xff;


      if (key != 27)
      {
          if (!cv::imwrite(output_name, output))
          {
              std::cerr << "Error: could not save the result in file '"<<output_name<<"'."<< std::endl;
                return EXIT_FAILURE;
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


