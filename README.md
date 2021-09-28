# FSIV-2122
Directorio de la asignatura de Fundamentos de Sistemas en Vision
# COMPILACION DE UNA PRACTICA USANDO QTCREATOR
  1) Abrimos qtCreator y seleccionamos <Welcome -> Open -> Open Creator>
  2) Navegamos hasta la carpeta de practicas y seleccionamos el fichero <CMakeList.txt>
  3) Configuramos el proyecto ( se recomienda usar solo las configuraciones <Release,Debug> )
  4) Pulsamos sobre el icono <martillo> para compilar el proyecto

 # COMPILACION DE UNA PRACTICA USANDO TERMINAL
  1) Accedemos a la carpeta del proyecto
  2) Creamos la carpeta build -> mkdir build
  3) Creamos el proyect -> cd build; cmake ..
  4) Ejecutamos el fichero makefile -> make
  5) Ejecutamos los archivos ejecutables -> Por ejemplo: ./ejecutable ../data/imagen.png
  
# FUNCIONES UTILIZADAS EN PRACTICAS
  
  1) Crear un gestor de la linea de comandos -> cv::ComandLineParser(argc,argv,keys)
  2) Indicar el nombre del programa -> parser.about(nombre_programa)
  3) Cargar una imagen
    3.1) cv::Mat img = cv::imread (nombre_imagen, cv::IMREAD_ANYCOLOR)
    3.2) cv::Mat img = cv::imread (nombre_imagen, cv::IMREAD_GRAYSCALE)
    3.3) cv::Mat img = cv::imread (nombre_imagen, cv::IMREAD_COLOR)
