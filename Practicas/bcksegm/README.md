# Segmentación del fondo en vídeo.

Se han realizado todas las parte de la practica

· fsiv_remove_segmentation_noise
· fsiv_segm_by_dif
· fsiv_apply_mask
· vsegbase.cpp
· fsiv_learn_gaussian_model
· fsiv_segm_by_gaussian_model
· fsiv_update_gaussian_model
· vsegadv.cpp
· Los programas permiten cambiar los parámetros de forma interactiva


./vsegbase -t=12 -s=1 -v=../data/campus_000_002.avi video.mp4
./vsegbase -t=12 -v=../data/campus_000_002.avi video.mp4
./vsegbase -s=1 -v=../data/campus_000_002.avi video.mp4
./vsegbase -t=12 -s=1 -c=/dev/video0 video.mp4


./vsegadv -b=100 -a=0.01 -k=0.13 -u=25 -U=10 -v=../data/campus_000_002.avi video.mp4
./vsegadv -b=100 -a=0.01 -k=0.13 -r=1 -g=15 -v=../data/campus_000_002.avi video.mp4
./vsegadv -b=100 -a=0.01 -k=0.13 -r=1 -g=15 -u=25 -U=10 -v=../data/campus_000_002.avi video.mp4
./vsegadv -b=100 -a=0.01 -c=/dev/video0 video.mp4


