# Calibración de la cámara con OpenCV.

Se han realizado todas las parte de la practica

· fsiv_generate_3d_calibration_points
· fsiv_find_chessboard_corners
· fsiv_calibrate_camera
· fsiv_compute_camera_pose
· fsiv_draw_axes
· fsiv_save_calibration_parameters
· fsiv_load_calibration_parameters
· fsiv_undistort_image
· fsiv_undistort_video_stream


./calibrate -s=0.04 -r=5 -c=6 -i=../data/logitech.xml ../data/parametrosIntrinsecos.xml ../data/logitech_000_004.png
./calibrate -s=0.04 -r=5 -c=6 -verbose -i=../data/logitech.xml ../data/parametrosIntrinsecos.xml ../data/logitech_000_008.png

./calibrate -s=0.04 -r=5 -c=6 ../data/parametrosIntrinsecos.xml ../data/logitech_000_001.png ../data/logitech_000_002.png ../data/logitech_000_003.png ../data/logitech_000_004.png ../data/logitech_000_005.png ../data/logitech_000_006.png ../data/logitech_000_007.png ../data/logitech_000_008.png

./calibrate -s=0.04 -r=5 -c=6 -verbose ../data/parametrosIntrinsecos.xml ../data/logitech_000_001.png ../data/logitech_000_002.png ../data/logitech_000_003.png ../data/logitech_000_004.png ../data/logitech_000_005.png ../data/logitech_000_006.png ../data/logitech_000_007.png ../data/logitech_000_008.png

./calibrate -s=0.04 -r=5 -c=6 -verbose -i=../data/parametrosIntrinsecos.xml ../data/parametrosIntrinsecos.xml ../data/logitech_000_004.png
./undistort ../data/elp_hd/elp-intrinsics.xml ../data/elp_hd/elp-view-002.jpg ../data/output.jpg
./undistort ../data/elp_hd/elp-intrinsics.xml ../data/elp_hd/elp-view-006.jpg ../data/output.jpg
./undistort -v -fourcc=MJPG ../data/logitech.xml ../data/tablero_000_000.avi ../data/outputVideo.avi
