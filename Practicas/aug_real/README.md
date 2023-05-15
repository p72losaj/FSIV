# Estimación de la Pose para Realidad Aumentada

Se han realizado todas las partes de la práctica:

· fsiv_generate_3d_calibration_points
· fsiv_find_chessboard_corners
· fsiv_compute_camera_pose
· fsiv_draw_axes
· fsiv_load_calibration_parameters
· fsiv_project_image
· Se dibuja un modelo 3D
· Se proyecta un vídeo.

Formas de ejecución

./aug_real 5 6 0.04 ../data/logitech.xml ../data/tablero_000_000.avi
./aug_real -m  5 6 0.04 ../data/logitech.xml ../data/tablero_000_000.avi
./aug_real -i=../data/computer-vision.jpg  5 6 0.04 ../data/logitech.xml ../data/tablero_000_000.avi
./aug_real -v=../data/tablero_000_000.avi 5 6 0.04 ../data/logitech.xml ../data/tablero_000_000.avi
