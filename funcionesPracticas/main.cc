/*!
	@brief Funcion principal para comprobacion de las practicas
	@author Jaime Lorenzo Sanchez
*/	

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

const cv::String keys =
    "{help h usage ? |      | print this message.   }"
    "{@image         |<none>| input image.          }" 
	"{@image2        |<none>| image2 for compare   }"
	"{documentacion opencv | | http://rabinf24.uco.es/fsiv/opencv-doc-3.4.4/html/index.html}"       
	"{w wait         |67    | number of msecs to wait between frames.}"
    "{camera c       |-1    | open camera index.}"
    "{video v        |      | open video source.}"    
    ;

int main (int argc, char* const* argv){
	int opcion = -1;

	try{
		cv::CommandLineParser parser(argc, argv, keys);
		if(parser.has("help")){
			parser.printMessage();
			return EXIT_FAILURE;
		}
		while(opcion != 0){
			mostrarMenu();
			std::cin >> opcion;
			// compute stats
			if(opcion == 1) {
				cv::String img_name = parser.get<cv::String>("@image");
				if (!parser.check())
      			{
          			parser.printErrors();
          			return EXIT_FAILURE;
      			}
				cv::Mat img = cv::imread(img_name, cv::IMREAD_ANYCOLOR); // Cargamos la imagen
    			if (img.empty())
    			{
        			std::cerr << "Error: Se ha producido un error al cargar la imagen " << img_name << std::endl;
        			return EXIT_FAILURE;
    			}
				comp_stats(img);
			}
			// esqueleto
			else if(opcion == 2) {
				cv::String img_name = parser.get<cv::String>("@image");
				if (!parser.check())
      			{
          			parser.printErrors();
          			return EXIT_FAILURE;
      			}
				cv::Mat img = cv::imread(img_name, cv::IMREAD_ANYCOLOR); // Cargamos la imagen
    			if (img.empty())
    			{
        			std::cerr << "Error: Se ha producido un error al cargar la imagen " << img_name << std::endl;
        			return EXIT_FAILURE;
    			}
				esqueleto(img);
			}
			// show image
			else if(opcion == 3) {
				cv::String img_name = parser.get<cv::String>("@image");
				if (!parser.check())
      			{
          			parser.printErrors();
          			return EXIT_FAILURE;
      			}
				cv::Mat img = cv::imread(img_name, cv::IMREAD_ANYCOLOR); // Cargamos la imagen
    			if (img.empty())
    			{
        			std::cerr << "Error: Se ha producido un error al cargar la imagen " << img_name << std::endl;
        			return EXIT_FAILURE;
    			}
				show_image(img);
			}
			// show video
			else if(opcion == 4) {
				int wait = parser.get<int>("w");      
      			int camera_idx = parser.get<int>("camera");
      			std::string video_name = parser.get<std::string>("video");
				if (!parser.check())
      			{
          			parser.printErrors();
          			return EXIT_FAILURE;
      			}
				int opcion_video = 0;
				if (parser.has("video")) opcion_video = 1;
				show_video(video_name,opcion_video, camera_idx,wait);
				
			}
		}
	}catch (std::exception& e)
  	{
    	std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    	return EXIT_FAILURE;
  	}
	return EXIT_SUCCESS;
}