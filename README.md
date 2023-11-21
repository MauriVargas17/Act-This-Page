# Act This Page

### Nombre: Mauricio Vargas Escobar

### Codigo: 56224

## Instrucciones

Se debe levantar el servidor de FastAPI ejecutando el archivo app.py ubicado en la carpeta backend. Se uso python 3.11 para el desarrollo.

Para acceder al cliente, abrimos el archivo basic_app.html de la carpeta frontend en un navegador.

Todo listo para jugar!

## Descripcion

Esta es una aplicacion que se presenta como un videojuego, el cual tiene el objetivo de mostrar una imagen famosa de alguna pelicula esperando que el usuario pueda imitar la imagen, tanto la pose como los elementos, para que mediante un calculo de los embeddings o vectores de datos, se pueda comparar la similitud de las imagenes y generar un score.

Se hace un pre procesamiento de las imagenes, tanto de las generadas por la web cam en el navegador como de las imagenes de peliculas. Luego se envian ambas imagenes al modelo para obtener los resultados.

## Disclaimer

A traves del cliente basic_app se puede hacer uso de la api, especificamente de la ruta /predict. Las rutas /status y /report no estan implementadas en el frontend pero pueden ser accedidas desde la url del servidor activo en http//localhost:8001/docs
