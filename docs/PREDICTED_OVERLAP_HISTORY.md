# Historial del experimento de solapamiento predicho

Este historial cubre los commits de la rama `experiment/predicted-overlap-fmap`.
El baseline se entrenó por separado. Todas las comparaciones de esta rama usan
pesos congelados y el checkpoint seleccionado por el error `point` de validación.

| Commit | Cambio | Justificación |
| --- | --- | --- |
| `9706f0b` | Añade `fmap_predicted` y una configuración que exige un checkpoint explícito. | Compara la ponderación suave de los descriptores de Y con `point`, `fmap` y el oracle, usando los mismos pesos y mapas. La exigencia del checkpoint evita evaluar por accidente con otro modelo. |
| `4e47076` | Añade el lanzador de evaluación para Marvin. | Permite enviar la evaluación con una ruta a `best.pth`, crea el override y registra el hash del checkpoint y el commit evaluado. |
| `8bf228f` | Reduce el límite de SLURM de dos horas a diez minutos. | La primera evaluación oficial tardó 1 min 35 s. Un límite más corto facilita que Slurm encuentre un hueco, con margen para la ejecución. |
| `da27ece` | Añade `fmap_predicted_binary` con umbral fijo `> 0,5` para el split de validación. | Comprueba si el uso de pesos continuos explica el resultado observado. El lanzador verifica los 32 mapas contra el manifiesto de validación. |
| `Export paired overlap evaluation errors` | Exporta los errores por dirección y por par sin orden, con diferencias respecto a `fmap`. | Permite comprobar si el cambio de signo entre validación y test se concentra en pocos pares o aparece de forma generalizada. |
| `Export vertex diagnostics for the horse–hippo pair` | Exporta probabilidades, GT y errores por vértice en ambas direcciones del par elegido. | El oracle mejora el par en test mientras la máscara suave lo empeora. El diagnóstico permite localizar los vértices que más contribuyen al cambio. |
| `Isolate the effect of missed overlap vertices` | Añade `fmap_gt_restore_fn` solo para las dos direcciones del par elegido. En la ponderación suave, restaura a 1 los vértices GT con probabilidad ≤ 0,5. | Mide el efecto de corregir exclusivamente los falsos negativos con los pesos congelados. El promedio de esta variante cubre el par de diagnóstico y no todo el test. |
| `Evaluate the fixed binary overlap mask on official test` | Añade `--binary-test` al lanzador para evaluar la máscara binaria con el umbral de validación `> 0,5`. Usa una carpeta de resultados nueva y desactiva el rescate GT limitado a un par. | Completa la comparación con las cinco variantes sobre los mismos 100 mapas del test oficial y el mismo checkpoint congelado. |
| `Visualize local correspondences for the horse–hippo pair` | Guarda los índices de destino de cada variante y añade el modo `--visualize-pair` para generar un visor HTML por dirección. | Permite comparar el destino de un vértice y los de sus vecinos de malla para estudiar dónde se pierde la coherencia local. |

## Cómo leer los CSV

La evaluación guarda ambos archivos en su directorio de resultados:

- `variant_errors_by_direction.csv` identifica el mapa como `source_y → target_x`.
  La máscara, cuando se usa, se aplica a los descriptores de la fuente Y.
- `variant_errors_by_pair.csv` reúne las direcciones del mismo par de formas.
  Pondera cada dirección por su número de correspondencias evaluadas, como
  en la agregación de errores del evaluador. La columna `directions` indica
  cuántas direcciones contribuyeron.
- `delta_vs_fmap` es el error de la variante menos el de `fmap`.
  Un delta negativo indica una mejora. Para comparar splits, filtra por
  `fmap_predicted` y `fmap_oracle`, disponibles en ambos. Cuenta los pares
  que mejoran y revisa los deltas más extremos en cada dirección.

Los valores agregados que imprime el evaluador se siguen calculando sobre
todas las correspondencias. Los CSV permiten contar pares con el mismo peso
si la pregunta es en cuántos pares mejora una variante.
