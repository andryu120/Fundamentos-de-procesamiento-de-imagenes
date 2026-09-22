# Pregunta 1: Saturación Selectiva de Color

Este directorio contiene la implementación de la Pregunta 1

## Estructura de Archivos
* **`espacios_color.py`:** Contiene todas las transformaciones entre espacios de color (RGB a HSV, RGB a CIE L*C*h* y sus inversas).
* **`procesamiento.py`:** Acá se encuentran las funciones que procesan y utilizan la información necesaria (interpolación, funciones de transformación, etc.).
* **`graficos.py`:** Contiene el código encargado de generar los gráficos de comparación de saturación utilizados en el informe.
* **`main.py`:** Este archivo se encarga de la ejecución principal del programa.

## Instrucciones de Ejecución y Reproducción de Resultados

Para generar las imágenes modificadas que se analizan en el  informe, ejecute:
```bash
python main.py
```
Los resultados se guardarán automáticamente en los directorios imagenes_modificadas_cie e imagenes_modificadas_hsv

Para generar los graficos mostrados en el Informe, ejecute:
```bash
python graficos.py
```
Para las pruebas de mostrar pixeles y modificar un punto p cualquiera, hay que ir a **`main.py`:** y descomentar las lineas del codigo que se quiere probar
