# Ejercicio 3: Aprende una función sinuidal con PyTorch

## Objetivo

Estimación de una función desconocida mediante un modelo de aprendizaje automático

## Formalización de tareas

La tarea que se puede formalizar en dos pasos. Primero, definiremos lo que intentamos lograr de la forma más clara posible. En segundo lugar, definiremos el enfoque que estamos adoptando para resolverlo.

### Formalización de tareas (Inferencia)

Existe una función desconocida $f$ para lo cual disponemos de un montón de datos sobre ciertas entradas $x$ y su salida correspondiente $y$.

$$
y = f(x)
$$

Estamos intentando crear un modelo de $f$ usando un método de aprendizaje automático para inferir la matriz de pesos de $W$ que exprese mejor la relación entre $x$-$y$ de datos. Expresado matemáticamente:

$$
y = f(W,x)
$$

Expresado gráficamente:

```mermaid
graph TD
    A((x)) --> B["f(W,x)"]
    B --> C((y))
    
```

El vector de entrada tiene tamaño [bs x 1]. La matriz de pesos tiene un tamaño [1 x 1]

### Formalización de tareas (Entrenamiento)

#### Explicación del diagrama de entrenamiento

El diagrama representa el proceso completo de entrenamiento de un modelo de
Machine Learning entrenado mediante la minimización de una función de pérdida.

##### Elementos del diagrama

- x: vector de entrada del modelo (datos de entrada).
- y: valor real u objetivo.
- f(W, x): modelo o función parametrizada por los pesos W, que genera
  una predicción a partir de la entrada.
- y′: salida estimada o predicción del modelo.
- Loss: función de pérdida que mide la diferencia entre la predicción
  y′ y el valor real y.
- W: conjunto de parámetros (pesos sinápticos) del modelo que se ajustan
  durante el entrenamiento.

##### Flujo del proceso de aprendizaje

1. La entrada **x** se introduce en el modelo **f(W, x)**.
2. El modelo genera una predicción **y′**.
3. La predicción **y′** se compara con el valor real **y** mediante la función
   de pérdida **Loss(y, y′)**.
4. La función de pérdida produce un valor escalar que cuantifica el error del
   modelo.
5. Este error se utiliza para actualizar los pesos **W**, generalmente mediante
   un algoritmo de optimización basado en gradiente descendente.
6. Los pesos actualizados se realimentan al modelo, cerrando el ciclo de
   entrenamiento.

Este proceso se repite de forma iterativa hasta que la función de pérdida
converge o se alcanza un criterio de parada.

```mermaid
graph TD
    A((x))
    B((y))
    M["Modelo f(W,x)"]
    C((y'))
    L["Loss(y, y')"]
    O["Optimizador (Gradiente Descente o Adam)"]
    W((W))

    A --> M
    W --> M
    M --> C
    C --> L
    B --> L
    L --> O
    O --> W
```

## Métricas de evaluación

Como estamos tratando con un problema de regresión, utilizaremos el error cuadrático medio (MSE), el error absoluto medio (MAE) y el R-cuadrado como métricas de evaluación.

## Consideraciones de datos

### Descripción del conjunto de datos

El conjunto de datos contiene 10000 puntos de datos ruidosos con una desviación estándar de ruido de 20 respecto a la función real ($y = 100 · sin(8 · π · x / 100) + 2$).

### Preparación y preprocesamiento de datos

No se ha realizado ningún preprocesado. El conjunto de datos se ha dividido en conjuntos de entrenamiento, validación y prueba.

### Aumento de datos

No se ha realizado ninguna ampliación de datos.

## Consideraciones del modelo

Para la función $y = 100 · sin(8 · π · x / 100) + 2$, un modelo lineal (SinglePerceptron) no es suficiente, ya que solo puede representar relaciones lineales entre la entrada y la salida.

Por este motivo se utiliza un perceptrón multicapa (MultiLayerPerceptron) con dos capas ocultas (fc1 y fc2) y función de activación ReLU. Esta arquitectura permite introducir no linealidad en el modelo y capturar la forma sinusoidal de la función objetivo. En la última capa se utiliza una activación identidad, ya que se trata de una tarea de regresión y la salida no debe estar limitada a un rango concreto.

### Funciones de pérdida adecuadas

La función de pérdida utilizada depende del tipo de problema:

- En regresión no lineal, se emplea el error cuadrático medio (MSE).
- En problemas de clasificación, se utiliza la entropía cruzada.

### Función de Pérdida Seleccionada

En esta tarea se utiliza la función de pérdida MSE (Mean Squared Error), ya que el problema planteado es un problema de regresión no lineal, en el que la salida del modelo es una variable continua.

La función MSE mide el error medio al cuadrado entre el valor real y y la predicción del modelo y′, y se define como:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - y'_i)^2$$

El objetivo del entrenamiento es ajustar los pesos sinápticos del modelo para minimizar esta función de pérdida mediante un algoritmo de optimización basado en gradiente descendente.

### Posibles arquitecturas [CAMBIARLO]

Se utiliza una arquitectura de perceptrón multicapa (MultilayerPerceptron) con 64 neuronas en la capa oculta. Esta arquitectura tiene múltiples parámetros (pesos y sesgos en fc1 y fc2) que se aprenden durante el entrenamiento: $W_1, b_1$ en la capa oculta y $W_2, b_2$ en la capa de salida.

### Activación de la última capa

Como es una tarea de regresión sin límites inferiores ni superiores, la activación de la última capa se establece en función Identidad.

### Otras consideraciones [CAMBIARLO]

Se usa `AdamW` como optimizador por su estabilidad y capacidad de convergencia. Para evitar saturar la salida, la última capa se deja sin activación (`Identity`).

## Entrenamiento [CAMBIARLO]

El entrenamiento se ha realizado a lo largo de 200 épocas. El gráfico de la función de pérdida se muestra a continuación.

### Hiperparámetros de entrenamiento [CAMBIARLO]

La tasa de aprendizaje se establece en 0,0001

### Grafo de la función de pérdida

![image](../../outs/exercise_03/loss_plot.png)

### Discusión sobre el proceso de entrenamiento [CAMBIARLO]

El loss de entrenamiento y validación disminuye rápidamente en las primeras épocas y luego se estabiliza. Las curvas se mantienen muy próximas, lo que indica buena generalización y ausencia de sobreajuste. El punto óptimo se alcanza alrededor de 200 épocas, donde la validación no mejora significativamente.

## Evaluación

### Métricas de evaluacións

Escribe tu respuesta aquí.

![image](../../outs/exercise_03/train_regression_plot.png)

![image](../../outs/exercise_03/validation_regression_plot.png)

![image](../../outs/exercise_03/test_regression_plot.png)

Las métricas de cada conjunto de datos se representan:

![image](../../outs/exercise_03/metrics.png)

### Evaluación de los resultados

Aquí tenéis ejemplos de resultados de evaluación para conjuntos de entrenamiento, validación y prueba.

Ejemplo para el conjunto de entrenamiento:

![image](../../outs/exercise_03/train_data_points_plot.png)

Ejemplo para el conjunto de validación:

![image](../../outs/exercise_03/validation_data_points_plot.png)

Ejemplo para el conjunto de pruebas:

![image](../../outs/exercise_03/test_data_points_plot.png)

### Discusión de los resultados

¿Cómo resuelve el modelo el problema?
¿Hay sobreajuste, subajuste o algún otro problema? 
¿Cómo podemos mejorar el modelo?
¿Cómo se generalizará este modelo a nuevos datos?

## Diseño de bucles de retroalimentación

Describe el proceso que has seguido para mejorar el modelo y la evolución del rendimiento del modelo durante el proceso.

Puedes incluir una tabla que indique los chanched parameters y los resultados obtenidos tras el proceso.

## Preguntas

Por favor, responde a las siguientes preguntas. Incluye gráficos si es necesario. Almacenar los gráficos en la carpeta `outs/exercise_03`.

### ¿Cuáles son las diferencias que encontraste entre el modelo anterior y este?

### ¿El modelo se generaliza bien a datos nuevos?



    # con una relu y a 400 epocas el best validation es a 3711
    # con dos relu y a 200 epocas el best validation es a 3294
    # con dos relu, 128 neuronas y a 200 epocas el best validation es a 3294
    # si está normalizada el validation loss baja a 2425 con 200 epocas y 64 neuronas
    # No se van a poner 128 neuronas porque no merece la pena
    # 2415 con 300 epocas
    # 2398 con 400 epocas
    # 672 con 400 epocas y 128 neuronas
    # 672 con 500 epocas y 128 neuronas

    # las 128 neuronas están en fila