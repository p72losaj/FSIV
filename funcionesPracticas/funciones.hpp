#ifndef FUNCIONES_HPP
#define FUNCIONES_HPP
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*!
    @brief Funcion que muestra un menu de opciones
*/
void mostrarMenu();
/*!
    @brief Funcion que muestra un video
    @param video_name Nombre del video
    @param opcion_video comprobacion de existencia del video
    @param camera_idx Opcion de camara
    @param wait Tiempo de espera entre frames en ms
*/
void show_video(const std::string video_name, const int opcion_video, const int camera_idx,const int wait);
/**
 * @brief Función callback para gestión del ratón.
 * @param event Qué ocurrió.
 * @param x coordenada x del cursor del ratón.
 * @param y coordenada y del cursor del ratón.
 * @param flags estado del teclado.
 * @param userdata datos que el usuario ha pasado al crear el callback.
 */
void on_mouse(int event, int x, int y, int flags, void *userdata);
/*!
    Funcion que muestra una imagen en una nueva ventana
*/
void show_image(const cv::Mat& img);
/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.
  Se supone que se utilizará OpenCV.
  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`
*/
void esqueleto(const cv::Mat& img);
/*!
    @brief Funcion que calcula los valores estadisticos de una imagen
    @param Imagen
*/
void comp_stats(const cv::Mat& img);
/*!
    @brief Calcular el valor medio de una imagen y su varianza.
    Esta es la forma más intuitiva de recorrer una imagen.
    @param[in] img es la imagen de entrada.
    @param[out] media la media de los valores.
    @param[out] dev la desviación estárdar de los valores.
    @pre img no está vacia.
    @pre img es de tipo CV_8UC1 (Un sólo canal en formato byte).
*/    
void  compute_stats1(const cv::Mat& img, float& media, float& dev);
/*!
    @brief Calcular el valor medio de una imagen y su varianza.
    Esta forma usa iteradores, más fácil de codificar cuando sólo queremos
    procesar todos los pixeles uno a uno.
    @param[in] img es la imagen de entrada.
    @param[out] media la media de los valores.
    @param[out] dev la desviación estárdar de los valores.
    @pre img no está vacia.
    @pre img es de tipo CV_8UC1 (Un sólo canal en formato byte).
*/
void
compute_stats2(const cv::Mat& img, float& media, float& dev);
/*!
    @brief Calcular el valor medio de una imagen y su varianza.

    Esta forma usa código vectorizado (funciones SIMD). Utilizamos funciones
    de opencv que permiten vectorizar el código.

    @param[in] img es la imagen de entrada.
    @param[out] media la media de los valores.
    @param[out] dev la desviación estárdar de los valores.

    @pre img no está vacia.
    @pre img es de tipo CV_32FC1 (Un sólo canal en formato float).
*/
void
compute_stats3(const cv::Mat& img, float& media, float& dev);
/*!
    @brief Calcular el valor medio de una imagen y su varianza.
    Esta forma usa una función de opencv (no siempre hay una!!).
    @param[in] img es la imagen de entrada.
    @param[out] media la media de los valores.
    @param[out] dev la desviación estárdar de los valores.
    @pre img no está vacia.
    @pre img es de tipo CV_32FC1 (Un sólo canal en formato float).
*/
void
compute_stats4(const cv::Mat& img, float& media, float& dev);

#endif