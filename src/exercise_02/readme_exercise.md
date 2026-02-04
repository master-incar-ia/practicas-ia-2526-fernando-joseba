# Ejercicio 2: Aprende una función lineal con PyTorch

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

El conjunto de datos contiene 10.000 puntos de datos ruidosos con una desviación estándar de ruido de 20 respecto a la función real ($y = -3x^2 + 5x$).

### Preparación y preprocesamiento de datos

No se ha realizado ningún preprocesado. El conjunto de datos se ha dividido en conjuntos de entrenamiento, validación y prueba.

### Aumento de datos

No se ha realizado ninguna ampliación de datos.

## Consideraciones del modelo

Para la función $y = -3x^2 + 5x$, un modelo lineal (SinglePerceptron) no es suficiente. Por ello utilizamos un perceptrón multicapa (MultilayerPerceptron) con **dos capas ocultas** (fc1 y fc2, ambas con 64 neuronas) y activación ReLU, y activación identidad en la última capa (fc3) para regresión sin límites. La arquitectura es: **1 → 64 (ReLU) → 64 (ReLU) → 1**. Esto permite capturar mejor la no linealidad de la función objetivo cuadrática.

### Funciones de pérdida adecuadas 

La función de pérdida utilizada depende del tipo de problema:

- En regresión no lineal, se emplea el error cuadrático medio (MSE).
- En problemas de clasificación, se utiliza la entropía cruzada.

### Función de Pérdida Seleccionada

En esta tarea se utiliza la función de pérdida MSE (Mean Squared Error), ya que el problema planteado es un problema de regresión no lineal, en el que la salida del modelo es una variable continua.

La función MSE mide el error medio al cuadrado entre el valor real y y la predicción del modelo y′, y se define como:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - y'_i)^2$$

El objetivo del entrenamiento es ajustar los pesos sinápticos del modelo para minimizar esta función de pérdida mediante un algoritmo de optimización basado en gradiente descendente.

### Posibles arquitecturas

Se utiliza una arquitectura de perceptrón multicapa (MultilayerPerceptron) con **dos capas ocultas de 64 neuronas cada una**. Esta arquitectura tiene múltiples parámetros (pesos y sesgos en fc1, fc2 y fc3) que se aprenden durante el entrenamiento: $W_1, b_1$ en la primera capa oculta, $W_2, b_2$ en la segunda capa oculta, y $W_3, b_3$ en la capa de salida.

### Activación de la última capa

Como es una tarea de regresión sin límites inferiores ni superiores, la activación de la última capa se establece en función Identidad.

### Otras consideraciones

Se usa `AdamW` como optimizador por su estabilidad y capacidad de convergencia. Para evitar saturar la salida, la última capa se deja sin activación (`Identity`).

## Entrenamiento

El entrenamiento se ha realizado a lo largo de **340 épocas**. El gráfico de la función de pérdida se muestra a continuación.

### Hiperparámetros de entrenamiento

- **Learning rate**: 0.0003
- **Batch size**: 64
- **Optimizador**: AdamW
- **Épocas**: 340

### Grafo de la función de pérdida

![image](../../outs/exercise_02/loss_plot.png)

### Discusión sobre el proceso de entrenamiento

El loss de entrenamiento y validación disminuye rápidamente en las primeras épocas y luego se estabiliza. Las curvas se mantienen muy próximas, lo que indica buena generalización y ausencia de sobreajuste. El punto óptimo se alcanza alrededor de 200 épocas, donde la validación no mejora significativamente.

## Evaluación

### Métricas de evaluación

Las métricas obtenidas muestran un rendimiento alto (que se puede observar en la tabla) en train/validation/test, y errores MAE/MSE consistentes entre conjuntos. Esto indica que el modelo explica la mayor parte de la variabilidad de los datos y generaliza correctamente.

![image](../../outs/exercise_02/train_regression_plot.png)

![image](../../outs/exercise_02/validation_regression_plot.png)

![image](../../outs/exercise_02/test_regression_plot.png)

Las métricas de cada conjunto de datos se representan:

![image](../../outs/exercise_02/metrics.png)

### Evaluación de los resultados

Imágenes de los resultados
Ejemplo para el conjunto de entrenamiento:

![image](../../outs/exercise_02/train_data_points_plot.png)

Ejemplo para el conjunto de validación:

![image](../../outs/exercise_02/validation_data_points_plot.png)

Ejemplo para el conjunto de pruebas:

![image](../../outs/exercise_02/test_data_points_plot.png)

### Discusión de los resultados

¿Cómo resuelve el modelo el problema?

El modelo aprende la forma cuadrática de la función objetivo y reproduce correctamente la curvatura en los datos.

¿Hay sobreajuste, subajuste o algún otro problema? 

 No se observa sobreajuste importante, ya que train/validation/test son muy similares.

¿Cómo podemos mejorar el modelo?

Para mejorar el modelo, se podría ajustar el número de neuronas o aplicar early stopping. Si se observa que el valor de Validation Test disminuye en las últimas épocas se podria subir el número de epocas.

¿Cómo se generalizará este modelo a nuevos datos?

  Dado el alto $R^2$, se espera una buena generalización a nuevos datos generados con la misma distribución.

## Diseño de bucles de retroalimentación

Describe el proceso que has seguido para mejorar el modelo y la evolución del rendimiento del modelo durante el proceso.

Estrategia seguida:
1. Añadir una segunda capa oculta (1→64→1 a 1→64→64→1) para aumentar la capacidad del modelo.
2. Ajustar el learning rate para estabilidad (0.001 → 0.0003) tras observar picos en validation loss.
3. Aumentar épocas progresivamente (100→200→300→340) hasta observar convergencia.
4. Mantener batch_size=64 (bueno para dataset de 10,000 puntos).

Puedes incluir una tabla que indique los parámetros cambiados y los resultados obtenidos tras el proceso.

| Cambio | Antes | Después | Impacto observado |
|---|---|---|---|
| Modelo | `SimplePerceptron` | `MultiLayerPerceptron` (3 capas) | Mejor ajuste de la curva (no linealidad) |
| Capas ocultas | 1 capa (64 neuronas) | 2 capas (64 + 64 neuronas) | Mayor capacidad para aproximar $y = -3x^2 + 5x$ |
| Learning rate | 0.001 (inestable) | 0.0003 | Convergencia más estable, sin picos |
| Épocas | 100 → 200 → 300 | 340 | Mejora del validation loss hasta estabilizar |

## Preguntas

Por favor, responde a las siguientes preguntas. Incluye gráficos si es necesario. Almacenar los gráficos en la carpeta `outs/exercise_02`.

### ¿Cuáles son las diferencias que encontraste entre el modelo anterior y este?

El modelo anterior era lineal (perceptrón simple, suficiente para la función anterior) y no podía aprender una relación cuadrática, produciendo un subajuste claro. El modelo actual es un perceptrón multicapa con activación no lineal, capaz de aproximar funciones no lineales, logrando un $R^2$ muy alto y errores menores.

### ¿El modelo se generaliza bien a datos nuevos?

Sí. Las métricas en train, validation y test son muy similares y el $R^2$ se mantiene alto en los tres conjuntos, lo que indica buena generalización.