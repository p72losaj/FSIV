#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "common_code.hpp"

std::vector<cv::Point3f>
fsiv_generate_3d_calibration_points(const cv::Size& board_size,
                                    float square_size)
{
    std::vector<cv::Point3f> ret_v;
    //TODO
    //Remenber: the first inner point has (1,1) in board coordinates.
    for(int i=1; i <= board_size.height; i++){
        for(int j=1; j<= board_size.width; j++){
            ret_v.push_back(cv::Point3f(j*square_size,i*square_size, 0));
        }
    }

    //
    CV_Assert(ret_v.size()==static_cast<size_t>(board_size.width*board_size.height));
    return ret_v;
}


bool
fsiv_find_chessboard_corners(const cv::Mat& img, const cv::Size &board_size,
                             std::vector<cv::Point2f>& corner_points,
                             const char * wname)
{
    CV_Assert(img.type()==CV_8UC3);
    bool was_found = false;
    //TODO
    // Buscamos las esquinas del tablero
    was_found = cv::findChessboardCorners(img,board_size,corner_points);
    // Esquinas encontrados -> Funcion cornerSubPix()
    if(was_found){
        cv::Mat grey = img.clone();
        cv::cvtColor(grey, grey, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(grey, corner_points, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria());
    }
    // Fichero de salida para mostrar el tablero -> Funcion drawChessboardCorners
    if(wname){
        cv::drawChessboardCorners(img, board_size, corner_points, was_found);
        cv::imshow(wname, img);
    }


    //
    return was_found;
}

float
fsiv_calibrate_camera(const std::vector<std::vector<cv::Point2f>>& _2d_points,
                      const std::vector<std::vector<cv::Point3f>>& _3d_points,
                      const cv::Size &camera_size,
                      cv::Mat& camera_matrix,
                      cv::Mat& dist_coeffs,
                      std::vector<cv::Mat>* rvecs,
                      std::vector<cv::Mat>* tvecs)
{
    CV_Assert(_3d_points.size()>=2 && _3d_points.size()==_2d_points.size());
    float error=0.0;
    //TODO
    if( (rvecs != nullptr) && (tvecs != nullptr)){
        // Obtenemos el error al calibrar la imagen
        error = cv::calibrateCamera(_3d_points, _2d_points, camera_size, camera_matrix, dist_coeffs, *rvecs, *tvecs);
    }
    else{
        std::vector<cv::Mat> rvecs_, tvecs_;
        error = cv::calibrateCamera(_3d_points, _2d_points, camera_size, camera_matrix, dist_coeffs, rvecs_, tvecs_);

    }


    //
    CV_Assert(camera_matrix.rows==camera_matrix.cols &&
              camera_matrix.rows == 3 &&
              camera_matrix.type()==CV_64FC1);
    CV_Assert((dist_coeffs.rows*dist_coeffs.cols) == 5 &&
              dist_coeffs.type()==CV_64FC1);
    CV_Assert(rvecs==nullptr || rvecs->size()==_2d_points.size());
    CV_Assert(tvecs==nullptr || tvecs->size()==_2d_points.size());
    return error;
}

void fsiv_compute_camera_pose(const std::vector<cv::Point3f> &_3dpoints,
                              const std::vector<cv::Point2f> &_2dpoints,
                              const cv::Mat& camera_matrix,
                              const cv::Mat& dist_coeffs,
                              cv::Mat& rvec,
                              cv::Mat& tvec)
{
    CV_Assert(_3dpoints.size()>=4 && _3dpoints.size()==_2dpoints.size());
    //TODO
        cv::solvePnP(_3dpoints, _2dpoints, camera_matrix, dist_coeffs, rvec, tvec);

    //
    CV_Assert(rvec.rows==3 && rvec.cols==1 && rvec.type()==CV_64FC1);
    CV_Assert(tvec.rows==3 && tvec.cols==1 && tvec.type()==CV_64FC1);
}

void
fsiv_draw_axes(cv::Mat& img,               
               const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
               const cv::Mat& rvec, const cv::Mat& tvec,
               const float size, const int line_width)
{
    //TODO
    // Vectores de puntos
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    // Obtenemos los puntos 3D
    cv::Point3f p0(0, 0, 0);
    cv::Point3f p1(size, 0, 0);
    cv::Point3f p2(0, size, 0);
    cv::Point3f p3(0, 0, -size);
    // Almacenamos los puntos 3D
    points3d.push_back(p0);
    points3d.push_back(p1);
    points3d.push_back(p2);
    points3d.push_back(p3);
    // Realizamos la proyección de los puntos 3D
    cv::projectPoints(points3d, rvec, tvec, camera_matrix, dist_coeffs, points2d);
    // Obtenemos la linea de puntos 2D
    cv::line(img, points2d[0], points2d[1], cv::Scalar(0, 0, 255), line_width);
    cv::line(img, points2d[0], points2d[2], cv::Scalar(0, 255, 0), line_width);
    cv::line(img, points2d[0], points2d[3], cv::Scalar(255, 0, 0), line_width);

    //
}

void
fsiv_save_calibration_parameters(cv::FileStorage& fs,
                                const cv::Size & camera_size,
                                float error,
                                const cv::Mat& camera_matrix,
                                const cv::Mat& dist_coeffs,
                                 const cv::Mat& rvec,
                                 const cv::Mat& tvec)
{
    CV_Assert(fs.isOpened());
    CV_Assert(camera_matrix.type()==CV_64FC1 && camera_matrix.rows==3 && camera_matrix.cols==3);
    CV_Assert(dist_coeffs.type()==CV_64FC1 && dist_coeffs.rows==1 && dist_coeffs.cols==5);
    CV_Assert(rvec.type()==CV_64FC1 && rvec.rows==3 && rvec.cols==1);
    CV_Assert(tvec.type()==CV_64FC1 && tvec.rows==3 && tvec.cols==1);
    //TODO

    fs.write("image-width", camera_size.width);
    fs.write("image-height", camera_size.height);
    fs.write("error", error);
    fs.write("camera-matrix", camera_matrix);
    fs.write("distortion-coefficients", dist_coeffs);
    fs.write("rvec", rvec);
    fs.write("tvec", tvec);

    //
    CV_Assert(fs.isOpened());
    return;
}

void
fsiv_load_calibration_parameters(cv::FileStorage &fs,
                                 cv::Size &camera_size,
                                 float& error,
                                 cv::Mat& camera_matrix,
                                 cv::Mat& dist_coeffs,
                                 cv::Mat& rvec,
                                 cv::Mat& tvec)
{
    CV_Assert(fs.isOpened());
    //TODO
    fs["image-width"] >> camera_size.width;
    fs["image-height"] >> camera_size.height;
    fs["error"] >> error;
    fs["camera-matrix"] >> camera_matrix;
    fs["distortion-coefficients"] >> dist_coeffs;
    fs["rvec"] >> rvec;
    fs["tvec"] >> tvec;

    //
    CV_Assert(fs.isOpened());
    CV_Assert(camera_matrix.type()==CV_64FC1 && camera_matrix.rows==3 && camera_matrix.cols==3);
    CV_Assert(dist_coeffs.type()==CV_64FC1 && dist_coeffs.rows==1 && dist_coeffs.cols==5);
    CV_Assert(rvec.type()==CV_64FC1 && rvec.rows==3 && rvec.cols==1);
    CV_Assert(tvec.type()==CV_64FC1 && tvec.rows==3 && tvec.cols==1);
    return;
}

void
fsiv_undistort_image(const cv::Mat& input, cv::Mat& output,
                     const cv::Mat& camera_matrix,
                     const cv::Mat& dist_coeffs)
{
    //TODO
    //Hint: use cv::undistort.
    output = input.clone();
    cv::undistort(input, output, camera_matrix, dist_coeffs);
    //
}

void
fsiv_undistort_video_stream(cv::VideoCapture&input_stream,
                            cv::VideoWriter& output_stream,
                            const cv::Mat& camera_matrix,
                            const cv::Mat& dist_coeffs,
                            const int interp,
                            const char * input_wname,
                            const char * output_wname,
                            double fps)
{
    CV_Assert(input_stream.isOpened());
    CV_Assert(output_stream.isOpened());
    //TODO
    //Hint: to speed up, first compute the transformation maps
    //(one time only at the beginning using cv::initUndistortRectifyMap)
    // and then only remap (cv::remap) the input frame with the computed maps.
    cv::Mat frame_in, frame_out;
        cv::Mat map1, map2;
        cv::Size size = cv::Size(input_stream.get(cv::CAP_PROP_FRAME_WIDTH), input_stream.get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), camera_matrix, size, CV_32FC1, map1, map2);
        input_stream >> frame_in;
        double retard = 1000/fps;

        while(!frame_in.empty()){

            cv::remap(frame_in, frame_out, map1, map2, interp);

            output_stream.write(frame_out);

            cv::imshow(input_wname, frame_in);
            cv::imshow(output_wname, frame_out);

            cv::waitKey(retard);
            input_stream >> frame_in;
        }

    //
    CV_Assert(input_stream.isOpened());
    CV_Assert(output_stream.isOpened());
}
