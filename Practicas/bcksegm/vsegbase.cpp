/* 
   (c) Fundamentos de Sistemas Inteligenties en Vision
   University of Cordoba, Spain
*/

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
//#include <unistd.h>
#include <ctype.h>
#include <cmath>


#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include "common_code.hpp"


/*
  Use CMake to compile
*/

const cv::String keys =
        "{help h usage ? |      | print this message   }"
        "{t threshold    |13    | Segmentation threshold.}"
        "{s              |0     | Radius of structural element. 0 means not remove.}"
        "{g              |0     | Radius of the gaussian kernel. 0 means not average.}"
        "{c              |      | Use this camera idx.}"
        "{v              |      | Use this video file.}"
        "{@output        |<none>| Path to output video.}"
        ;
class Parametros{

public:

    Parametros(int threshold, int ste_radius){

        this->threshold = threshold;
        this->ste_radius = ste_radius;
    }

    int threshold;
    int ste_radius;
};

void trackbar_callback_threshold(int threshold, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->threshold = threshold;
}

void trackbar_callback_ste_radius(int ste_radius, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->ste_radius = ste_radius;
}


int
main (int argc, char * const argv[])
{

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Background segmentation in video.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    cv::VideoCapture input;
    bool is_camera = false;
    if (parser.has("c"))
    {
        input.open(parser.get<int>("c"));
        is_camera=true;
    }
    else if (parser.has("v"))
    {
        input.open(parser.get<std::string>("v"));
    }
    else
    {
        std::cerr << "Error: Wrong CLI. one of 'v' or 'c' options must be specified."
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (! input.isOpened())
    {
        std::cerr << "Error: could not open the input video stream."
                  << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat prev_frame;
    bool was_Ok = input.read(prev_frame);
    if (!was_Ok)
    {
        std::cerr << "Error: could not read any image from the input stream.\n";
        return EXIT_FAILURE;
    }

    std::string output_path = parser.get<std::string>("@output");
    int threshold = parser.get<int>("threshold");
    int ste_radius = parser.get<int>("s");
    int g_radius = parser.get<int>("g");

    cv::VideoWriter output;
    double fps=25.0; //Default value for live video.
    //TODO
    //Open the output video stream.
    //If the input is a video file, get the fps from it.

    cv::Size camera_size(int(input.get(3)),int(input.get(4)));

        int fourcc = input.get(CV_CAP_PROP_FOURCC);

        if (parser.has("v"))
        {
            fps = input.get(CV_CAP_PROP_FPS);
        }

        output = cv::VideoWriter(output_path,fourcc,fps,camera_size);


    //

    if (!output.isOpened())
    {
        std::cerr << "Error: could not open the output stream ["
                  << output_path << "]." << std::endl;
        return EXIT_FAILURE;
    }

    cv::namedWindow("Input");
    cv::namedWindow("Masked Input");
    cv::namedWindow("Output");
    //TODO
    //Trackbars to window Output to control the parameters.

    Parametros parametros(threshold, ste_radius);
    cv::createTrackbar("threshold", "Output", &parametros.threshold, 256, trackbar_callback_threshold, &parametros);
    cv::createTrackbar("ste_radius", "Output", &parametros.ste_radius, 25, trackbar_callback_ste_radius, &parametros);

    //

    cv::Mat curr_frame, mask, masked_frame;
    int key = 0;
    while(was_Ok && key!=27)
    {
        //TODO
        //Process the input stream until achiving the last frame or press "ESC" key.
        //First get the curr frame and used the prev one to compute the mask.
        //Try adjust waitKey's time to approximate the fps of the stream.

        was_Ok = input.read(curr_frame);

                if(curr_frame.empty()){

                    break;
                }

                imshow("Input", curr_frame);
                fsiv_segm_by_dif(prev_frame, curr_frame, mask, parametros.threshold, parametros.ste_radius);
                imshow("Masked Input", mask);
                fsiv_apply_mask(curr_frame, mask, masked_frame);
                imshow("Output", masked_frame);
                prev_frame=curr_frame.clone();
                output.write(masked_frame);

                cv::waitKey(1000/fps);



        //
    }

    return EXIT_SUCCESS;
}
