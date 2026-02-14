# Exercise 8: Create a Deep autoencoder in PyTorch with CIFAR-10 dataset

## Objective

Develop a deep autoencoder model that can classify compress and decompress images from CIFAR-10 dataset

1) Create a encoder decoder architecture. For the encoder part, use same architecture than exercise_05
2) Setset the autoencoder loss
2) Use the encoder and decoder to train and evaluate a model to compress and decompress images
3) Create an evaluate.py file that evaluates the model and calculates and stores the evaluation metrics (which are the ones that are needed?)
4) Use the encoder of the autoencoder to train and evaluate a model to classify in CIFAR-10 dataset

## Formalización de tareas

Escribe tu respuesta aquí.

### Formalización de tareas (Inferencia)

Escribe tu respuesta aquí.

### Formalización de tareas (Entrenamiento)

Escribe tu respuesta aquí.

## Métricas de evaluación

Escribe tu respuesta aquí.

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

![image](../../outs/exercise_08/loss_plot.png)

### Discusión sobre el proceso de entrenamiento

Escribe tu respuesta aquí.

## Evaluación

### Métricas de evaluacións

Escribe tu respuesta aquí.

![image](../../outs/exercise_08/train_regression_plot.png)

![image](../../outs/exercise_08/validation_regression_plot.png)

![image](../../outs/exercise_08/test_regression_plot.png)

Las métricas de cada conjunto de datos se representan:

![image](../../outs/exercise_08/metrics.png)

### Evaluación de los resultados

Aquí tenéis ejemplos de resultados de evaluación para conjuntos de entrenamiento, validación y prueba.

Ejemplo para el conjunto de entrenamiento:

![image](../../outs/exercise_08/train_data_points_plot.png)

Ejemplo para el conjunto de validación:

![image](../../outs/exercise_08/validation_data_points_plot.png)

Ejemplo para el conjunto de pruebas:

![image](../../outs/exercise_08/test_data_points_plot.png)


### Discusión de los resultados

¿Cómo resuelve el modelo el problema?
¿Hay sobreajuste, subajuste o algún otro problema? 
¿Cómo podemos mejorar el modelo?
¿Cómo se generalizará este modelo a nuevos datos?

## Diseño de bucles de retroalimentación

Describe el proceso que has seguido para mejorar el modelo y la evolución del rendimiento del modelo durante el proceso.

Puedes incluir una tabla que indique los chanched parameters y los resultados obtenidos tras el proceso.

## Preguntas

Por favor, responde a las siguientes preguntas. Incluye gráficos si es necesario. Almacenar los gráficos en la carpeta `outs/exercise_08`.

### ¿Cuáles son las diferencias que encontraste entre el modelo anterior y este?

### ¿El modelo se generaliza bien a datos nuevos?