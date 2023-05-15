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
    "{verbose        |      | activate verbose mode.}"
    "{i intrinsics   |      | Calibrate only extrinsics parameters. Using intrinsics from given file (-i=intr-file).}"
    "{s size         |<none>| square size.}"
    "{r rows         |<none>| number of board's rows.}"
    "{c cols         |<none>| number of board's cols.}"
    "{@output        |<none>| filename for output intrinsics file.}"
    "{@input1        |<none>| first board's view.}"
    "{@input2        |      | second board's view.}"
    "{@inputn        |      | ... n-idx board's view.}"
    ;

int
main (int argc, char* const* argv)
{
    int retCode=EXIT_SUCCESS;

    try {        
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Calibrate the intrinsics parameters of a camera.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }
        float square_size = parser.get<float>("s");
        int rows = parser.get<int>("r");
        int cols = parser.get<int>("c");
        bool verbose = parser.has("verbose");
        std::string output_fname = parser.get<cv::String>("@output");
        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }

        //Get the inputs.
        //find the second argument without '-' at begin.
        int input = 1;
        bool found = false;
        while (input<argc && !found)
            found = argv[input++][0] != '-';
        CV_Assert(input<argc);
        std::vector<std::string> input_fnames;        
        for (; input<argc; ++input)
            input_fnames.push_back(std::string(argv[input]));

        //TODO

        cv::Mat input_img, output_img;
                cv::Mat camera_matrix;
                cv::Mat dist_coeffs;
                cv::Mat rvec, tvec;

                std::vector<cv::Mat> rvecs;
                std::vector<cv::Mat> tvecs;


                std::vector<cv::Point3f> _3dpoints;
                std::vector<cv::Point2f> _2dpoints;

                std::vector<std::vector<cv::Point3f>> wcs;
                std::vector<std::vector<cv::Point2f>> ccs;


                std::string input_fname = parser.get<cv::String>("i");
                float error;
                cv::FileStorage fs;
                cv::Size camera_size;
                cv::Size board_size(cols-1, rows-1);
                cv::namedWindow("INPUT");
                cv::namedWindow("OUTPUT");



        if (parser.has("i"))
        {
            //TODO
            //Make extrinsic calibration.
            //Remenber: only one view is needed.


            fs.open(input_fname, cv::FileStorage::Mode::READ);

            fsiv_load_calibration_parameters(fs, camera_size, error, camera_matrix, dist_coeffs, rvec, tvec);

            input_img = cv::imread(input_fnames[0]);
            cv::imshow ("INPUT", input_img);

            fsiv_find_chessboard_corners(input_img, board_size, _2dpoints, "OUTPUT");
            _3dpoints = fsiv_generate_3d_calibration_points(board_size, square_size);

            fsiv_compute_camera_pose(_3dpoints, _2dpoints, camera_matrix, dist_coeffs, rvec, tvec);

            fs.open(output_fname, cv::FileStorage::Mode::WRITE);
            fsiv_save_calibration_parameters(fs, camera_size, error, camera_matrix, dist_coeffs, rvec, tvec);

            int k = cv::waitKey(0)&0xff;

                    if (k!=27)
                    {

                    }



            //
            if (verbose)
            {
                //TODO
                //Show WCS axis.
                fsiv_draw_axes(input_img, camera_matrix, dist_coeffs, rvec, tvec, 0.1);

                cv::imshow ("OUTPUT", input_img);

                int k = cv::waitKey(0)&0xff;

                        if (k!=27)
                        {

                        }


                //
            }
        }
        else
        {
            //TODO
            //Make an intrisic calibration.
            //Remember: For each view (at least two) you must find the
            //chessboard to get the 3D -> 2D matches.
            _3dpoints = fsiv_generate_3d_calibration_points(board_size, square_size);

            for(int i = 0; i < (int)input_fnames.size(); i++){

                input_img = cv::imread(input_fnames[i]);
                cv::imshow ("INPUT", input_img);

                bool was_found = fsiv_find_chessboard_corners(input_img, board_size, _2dpoints, "OUTPUT");

                if(was_found){

                    wcs.push_back(_3dpoints);
                    ccs.push_back(_2dpoints);
                }

                int k = cv::waitKey(0)&0xff;

                        if (k!=27)
                        {

                        }
            }

            error = fsiv_calibrate_camera(ccs, wcs, input_img.size(), camera_matrix, dist_coeffs, &rvecs, &tvecs);


            //

            if (verbose)
            {
                //TODO
                //Show WCS axis on each pattern view.
                for(int i = 0; i < (int)input_fnames.size(); i++){

                    input_img = cv::imread(input_fnames[i]);

                    fsiv_draw_axes(input_img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1);

                    cv::imshow ("OUTPUT", input_img);

                    int k = cv::waitKey(0)&0xff;

                            if (k!=27)
                            {

                            }
                }


                //
            }
            fs.open(output_fname, cv::FileStorage::Mode::WRITE);
            fsiv_save_calibration_parameters(fs, camera_size, error, camera_matrix, dist_coeffs);

        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
