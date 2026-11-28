# Pregunta 2: Compresión HDR y recuperación de color

Este directorio contiene la implementación de la Pregunta 2

## Estructura de Archivos
* **`procesamiento.py`:** Funciones para cargar la imagen y calcular filtro homorfico/durand por canal
* **`main.py`:** Este archivo se encarga de la ejecución principal del programa y donde se generan las imagenes a analizar.
* **`Durand.py`:** Implementacion del metodo HDR Durand
* **`homorfico.py`:** Implementacion del metodo filtro homorfico
* **`reconstruccion_luminancia.py`:** Implementacion del primer metodo de reconstruccion usando la razon de luminancia
* **`reconstruccion_color.py`:** Implementacion del segundo metodo de reconstruccion usando el espacio CIElch y modificando el croma
* **`graficar.py`:** Funciones para visualizar los resultados.

## Instrucciones de Ejecución y Reproducción de Resultados


Para generar las imágenes modificadas que se analizan en el informe, ejecute:
```bash
python main.py
```
Los resultados se mostraran en pantalla y en la terminal si corresponde