# Pregunta 2: Deconvolución de Wiener y Tikhonov iterativo

Este directorio contiene la implementación de la Pregunta 2

## Estructura de Archivos
* **`procesamiento.py`:** Contiene la degradacion y modificacion de la PSF.
* **`main.py`:** Este archivo se encarga de la ejecución principal del programa y donde se generan las imagenes a analizar.
* **`Wiener.py`:** Este archivo tiene la implementacion del filtro Wiener.
* **`Tikhonov.py`:** Este archivo tiene la implementacion del filtro Tikhonov.

## Instrucciones de Ejecución y Reproducción de Resultados

Previamente se requiere de la libreria rawpy para cargar la imagen, para ello de la terminal:

```bash
pip install rawpy
```


Para generar las imágenes modificadas que se analizan en el informe, ejecute:
```bash
python main.py
```
Los resultados se mostraran en pantalla