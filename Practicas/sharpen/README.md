# Enhance an image using a sharpening filter.

PARTES QUE HE REALIZADO DE LA PRACTICA:

+Tests
+Funciones del main
+Procesado de imagenes a color con luma incluido
+Modo interactivo

Formas de ejecución

./sharpen ../data/radiografia.png radiografia_output.png
./sharpen ../data/radiografia.png radiografia_output.png -r1=30 -r2=50 
./sharpen ../data/radiografia.png radiografia_output.png -r1=20 -r2=50 
./sharpen ../data/radiografia.png radiografia_output.png -r1=40 -r2=50 -f=0
./sharpen ../data/radiografia.png radiografia_output.png -r1=40 -r2=50 -f=1 -c
./sharpen ../data/radiografia.png radiografia_output.png -r1=40 -r2=50 -f=2
./sharpen ../data/radiografia.png radiografia_output.png -i

./sharpen ../data/ciclista_original.jpg ciclista_output.jpg  
./sharpen ../data/ciclista_original.jpg ciclista_output.jpg  -r1=30 -r2=50 -l
./sharpen ../data/ciclista_original.jpg ciclista_output.jpg  -r1=20 -r2=50 -l
./sharpen ../data/ciclista_original.jpg ciclista_output.jpg  -r1=40 -r2=50 -f=0 -l
./sharpen ../data/ciclista_original.jpg ciclista_output.jpg  -r1=40 -r2=50 -f=1 -c -l
./sharpen ../data/ciclista_original.jpg ciclista_output.jpg  -i

