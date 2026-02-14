# Ejercicio 4: Crear un Modelo de Aprendizaje Profundo para la clasificación de imágenes en PyTorch con el conjunto de datos CIFAR-10

## Objetivo

Desarrollar un modelo que pueda clasificar imágenes del conjunto de datos CIFAR-10

Luego prueba un modelo con capas convolucionales. Crear un archivo evaluate.py que evalúe el modelo y calcule y almacene las métricas de evaluación, incluyendo una matriz de confusión

Compara este método con el anterior (ejercicio anterior) ¿Cuál es el efecto de la ampliación de datos?

La ampliación de datos introduce variaciones (Rotaciones, Traslaciones, Zoom, etc) para reducir el overfitting, ya que el modelo ve versiones diferentes de una misma imagen. Debe mostrar mejora sobre una red convolucional sin ampliación de datos.

Compara ambos métodos y comenta las diferencias

La diferencia es que una CNN sin ampliación de datos tiende a memorizar el conjunto de entrenamiento (train dataset), tiene peor precisión en el test y no es muy robusto a variaciones en la imagen.
Con ampliación de datos la CNN generaliza mejor, aprende patrones visuales y tiene menor sobreajuste.

## Formalización de tareas

Escribe tu respuesta aquí.

### Formalización de tareas (Inferencia)

La red convilucional debe aprender a asignar cada imagen del CIFAR-10 una de las 10 clases posibles.
$$ R^{32X32X3} --> {0, ..., 9} $$

Durante la inferencia, la red ya tiene todos sus parámetros entrenados, así que simplemente aplica esa función aprendida a las imágenes correspondientes.

Estamos intentado crear un modelo CNN usando una estructura VGGNet que reduce el número de parámetros en las capas de convolución y mejora el tiempo de entrenamiento.

Input (3x32x32) -> Conv(3→32) + ReLU -> MaxPool (↓ tamaño a 16x16) -> Conv(32→64) + ReLU -> MaxPool (↓ tamaño a 8x8)

Después se aplana y pasa por capas lineales:

-> Flatten -> Linear + ReLU -> Dropout -> Linear

La entrada al modelo de cada imagen $$ x ∈ R^32X32X3 $$ y cuando se procesa en lotes $$ x ∈ R^ {bsX32X32X3} $$ donde bs es el batch size.

La predicción del modelo es la clase con mayor probabilidad.

$$ y'=argmaxp_i $$

La función completa es $$ 𝑦
=𝑓(𝑊,𝑋) $$
pero ahora 𝑊 representa todos los pesos de todas las capas, no una matriz 1×1.


### Formalización de tareas (Entrenamiento)

Durante el entrenamiento el objetivo de la CNN es aprender una función que aproxime la relación entre las imágenes de entrada x y sus clases y. Para ello, el modelo ajusta los pesos y sesgos de las capas convolucionales y lineales minimizando la función de pérdida para problemas de clasificación (CrossEntropy). 

## Métricas de evaluación

Las métricas de evaluación que se usan son Acurracy, la matriz de confusión y la funcion de pérdida (CrossEntropy)

## Consideraciones de datos

### Descripción del conjunto de datos

Escribe tu respuesta aquí.

### Preparación y preprocesamiento de datos

Escribe tu respuesta aquí.

### Aumento de datos

Escribe tu respuesta aquí.

## Consideraciones del modelo

Escribe tu respuesta aquí.

### Funciones de pérdida adecuadas

Escribe tu respuesta aquí.

### Función de Pérdida Seleccionada

Escribe tu respuesta aquí.

### Posibles arquitecturas

Escribe tu respuesta aquí.

### Activación de la última capa

Escribe tu respuesta aquí.

### Otras consideraciones

Escribe tu respuesta aquí.

## Entrenamiento

Escribe tu respuesta aquí.

### Hiperparámetros de entrenamiento

Escribe tu respuesta aquí.

### Grafo de la función de pérdida

![image](../../outs/exercise_04/loss_plot.png)

### Discusión sobre el proceso de entrenamiento

Escribe tu respuesta aquí.

## Evaluación

### Métricas de evaluacións

Escribe tu respuesta aquí.


Las métricas de cada conjunto de datos se representan:

![image](../../outs/exercise_04/metrics.png)

### Evaluación de los resultados

Aquí tenéis ejemplos de resultados de evaluación para conjuntos de entrenamiento, validación y prueba.

Ejemplo para el conjunto de entrenamiento:

![image](../../outs/exercise_04/train_confusion_matrix.png)

Ejemplo para el conjunto de validación:

![image](../../outs/exercise_04/validation_confusion_matrix.png)

Ejemplo para el conjunto de pruebas:

![image](../../outs/exercise_04/test_confusion_matrix.png)

### Discusión de los resultados

¿Cómo resuelve el modelo el problema?
¿Hay sobreajuste, subajuste o algún otro problema? 
¿Cómo podemos mejorar el modelo?
¿Cómo se generalizará este modelo a nuevos datos?

## Diseño de bucles de retroalimentación

Describe el proceso que has seguido para mejorar el modelo y la evolución del rendimiento del modelo durante el proceso.

Puedes incluir una tabla que indique los chanched parameters y los resultados obtenidos tras el proceso.

## Preguntas

Por favor, responde a las siguientes preguntas. Incluye gráficos si es necesario. Almacenar los gráficos en la carpeta `outs/exercise_04`.

### ¿Cuáles son las diferencias que encontraste entre el modelo anterior y este?

### ¿El modelo se generaliza bien a datos nuevos?