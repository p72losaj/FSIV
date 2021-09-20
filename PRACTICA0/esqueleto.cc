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

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{path           |.     | path to file         }"
    "{fps            | -1.0 | fps for output video }"
    "{N count        |100   | count of objects     }"
    "{ts timestamp   |      | use time stamp       }"
    "{@image1        |      | image1 for compare   }"
    "{@image2        |<none>| image2 for compare   }"
    "{@repeat        |1     | number               }"
    ;

int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Application name v1.0.0");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }
      int N = parser.get<int>("N");
      double fps = parser.get<double>("fps");
      cv::String path = parser.get<cv::String>("path");
      bool use_time_stamp = parser.has("timestamp");
      cv::String img1 = parser.get<cv::String>("@image1");
      cv::String img2 = parser.get<cv::String>("@image2");
      int repeat = parser.get<int>("@repeat");
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

    /*Ahora toca que tu rellenes con lo que hay que hacer ...*/

     // Cargamos las imagenes desde archivo

      std::cout << "Cargando la imagen: " << img1 << std::endl;

      cv::Mat img = cv::imread(img1, cv::IMREAD_ANYCOLOR);
     // cv::Mat imgB = cv::imread(img2, cv::IMREAD_ANYCOLOR);

      // Comprobamos si se ha cargado la imagen1 correctamente

      if (img.empty())
      {
         std::cerr << "Error: no he podido abrir el fichero '" << img1 << "'." << std::endl;
         return EXIT_FAILURE;
      }

      // Creamos un vector de canales

      std::vector<cv::Mat> canales;

      //Guardamos cada canal como una matriz diferente. Lo enlazamos con la imagen1

      cv::split(img, canales);

      // Recorremos cada canal de la imagen1

      for(size_t i=0; i < canales.size(); i++){

          double min, max;

          cv::minMaxIdx(canales[i],&min,&max);

          std::cout << "Canal: " << i << std::endl;

          // Valor maximo del canal

         std::cout<<"Valor maximo del canal: " << max << std::endl;

          // Valor minimo del canal

         std::cout << "Valor minimo del canal: " << min << std::endl;

      }

    
  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  catch (...)
  {
    std::cerr << "Capturada excepcion desconocida!" << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
