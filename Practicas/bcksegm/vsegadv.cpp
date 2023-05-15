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
    "{help h usage ? |      | Print this message   }"
    "{verbose        |0     | Set verbose mode level.}"
    "{b bframes      |100   | Number of frames to estimate background model.}"
    "{a alpha        |0.01  | Weight to update the background model.}"
    "{k threshold    |0.05  | Segmentation threshold.}"
    "{u stup         |10    | Short time updating period. Value 0 means not update.}"
    "{U ltup         |100   | Long time updating period. Value 0 means not update.}"
    "{r              |0     | Radius to remove segmentation noise. Value 0 means not remove.}"
    "{g              |0.0   | Radius to do a previous Gaussian averaging of input frames. Value 0 means not average.}"
    "{c              |      | Use this camera idx.}"
    "{v              |      | Use this video file.}"
    "{@output        |<none>| Path to output video.}"
    ;

class Parametros{

public:

    Parametros(int threshold, int ste_radius, int bframes, int stup, int ltup, float alfa, int gauss_radius){

        this->thresholdf = threshold/100.0f;
        this->threshold = threshold;
        this->ste_radius = ste_radius;
        this->bframes = bframes;
        this->stup = stup;
        this->ltup = ltup;
        this->alfaf = alfa/100.0f;
        this->alfa = alfa;
        this->gauss_radius = gauss_radius;
    }

    int threshold;
    float thresholdf;
    int ste_radius;
    int bframes;
    int stup;
    int ltup;
    int alfa;
    float alfaf;
    int gauss_radius;

};

void trackbar_callback_threshold(int threshold, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->thresholdf = threshold/100.0f;
}

void trackbar_callback_ste_radius(int ste_radius, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->ste_radius = ste_radius;
}

void trackbar_callback_bframes(int bframes, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->bframes = bframes;
}

void trackbar_callback_stup(int stup, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->stup = stup;
}

void trackbar_callback_ltup(int ltup, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->ltup = ltup;
}

void trackbar_callback_alfa(int alfa, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->alfaf = alfa/100.0f;
}

void trackbar_callback_gauss_radius(int gauss_radius, void* parametros)
{
    Parametros * aux = (Parametros *)parametros;
    aux->gauss_radius = gauss_radius;
}


void
on_change_K(int v, void *user_data)
{
    std::cout << "Setting K= " << v/100.0 << std::endl;
}

void
on_change_alfa(int v, void *user_data)
{
    std::cout << "Setting Alfa= " << v/100.0 << std::endl;
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
    int verbose = parser.get<int>("verbose");
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

    cv::Mat frame;
    bool was_Ok = input.read(frame);
    if (!was_Ok)
    {
        std::cerr << "Error: could not read any image from the input stream.\n";
        return EXIT_FAILURE;
    }

    std::string output_path = parser.get<std::string>("@output");
    float K = parser.get<float>("k");
    int K_int = int(K*100.0f);
    int ste_radius = parser.get<int>("r");
    int gauss_radius = parser.get<int>("g");
    int stup = parser.get<int>("u");
    int ltup = parser.get<int>("U");
    int num_frames = parser.get<int>("b");
    float alfa = parser.get<float>("a");
    int alfa_int = alfa*100.0f;

    cv::VideoWriter output;
    double fps=25.0; //Default value for live video.
    //TODO
    //Open the output video stream.
    //If the input is a video file, get the fps from it.

    cv::Size camera_size(int(input.get(3)),int(input.get(4)));
    int fourcc = input.get(CV_CAP_PROP_FOURCC);
    if(parser.has("v")){
        fps = input.get(CV_CAP_PROP_FPS);
    }
    output = cv::VideoWriter(output_path, fourcc, fps, camera_size);

    //

    if (!output.isOpened())
    {
        std::cerr << "Error: could not open the output stream ["
                  << output_path << "]." << std::endl;
        return EXIT_FAILURE;
    }

    cv::namedWindow("Input"); //show the current input frame.
    cv::namedWindow("Masked Input"); //show the current frame masked
    cv::namedWindow("Output"); //show the segmentation (mask).
    cv::namedWindow("Background"); //show the model (mean image.)
    //TODO
    //Add trackbars to window Output to control the parameters.

    Parametros parametros(K_int, ste_radius, num_frames, stup, ltup, alfa_int, gauss_radius);
    cv::createTrackbar("threshold", "Output", &parametros.threshold, 256, trackbar_callback_threshold, &parametros);
    cv::createTrackbar("ste_radius", "Output", &parametros.ste_radius, 25, trackbar_callback_ste_radius, &parametros);
    cv::createTrackbar("num_frames", "Output", &parametros.bframes, 500, trackbar_callback_bframes, &parametros);
    cv::createTrackbar("stup", "Output", &parametros.stup, 25, trackbar_callback_stup, &parametros);
    cv::createTrackbar("ltup", "Output", &parametros.ltup, 500, trackbar_callback_ltup, &parametros);
    cv::createTrackbar("alfa", "Output", &parametros.alfa, 100, trackbar_callback_alfa, &parametros);
    cv::createTrackbar("gauss_radius", "Output", &parametros.gauss_radius, 25, trackbar_callback_gauss_radius, &parametros);

    //

    cv::Mat Mean, Variance;
    //TODO
    //First, to learn the gaussian model.
    fsiv_learn_gaussian_model(input, Mean, Variance, parametros.bframes, parametros.gauss_radius, "Masked Input");

    //

    cv::Mat frame_f;
    cv::Mat mask;
    cv::Mat masked_frame;
    int key = 0;
    int frame_count = num_frames;

    while(was_Ok && key!=27)
    {
        //TODO
        //Process the input stream until achiving the last frame or press "ESC" key.
        //Remember: convert captured frames to in float [0,1].
        //First get the curr frame and use the learned model to segment it (get the mask).
        //Second update the model using the frame count.
        //Third get the masked frame.
        //Fourth show the results.
        //Fifth write the mask into the output RGB stream.
        //Try adjust waitKey's time to approximate the fps of the stream.


        was_Ok = input.read(frame_f);

        if(frame_f.empty()){
            break;
        }
        cv::Mat frame_faux;
        frame_f.convertTo(frame_faux, CV_32F, 1/255.0);
        imshow("Input", frame_f);

        fsiv_segm_by_gaussian_model(frame_faux, mask, Mean, Variance, parametros.thresholdf, parametros.ste_radius);
        fsiv_update_gaussian_model(frame_faux, mask, frame_count, Mean, Variance, parametros.alfaf, parametros.stup, parametros.ltup);
        imshow("Output", mask);
        fsiv_apply_mask(frame_f, mask, masked_frame);
        imshow("Masked Input", masked_frame);
        imshow("Background", Mean);
        cv::waitKey(1000/fps);



        //
    }

    return EXIT_SUCCESS;
}
