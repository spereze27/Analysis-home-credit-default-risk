### Analisis de riesgo de credito 
A continuacion se presenta el desarrollo de un modelo de predicción de capacidad de pago de creditos de los clientes basado en diferentes datos suministrados por Home Credit en Kaggle.
Para este ejercicio se trabajara sobre un subconjunto de datos que contiene los datos de la tabla train/test (tabla que contiene toda la informacion general), installments_payments que contiene informacion del comportamiento financiero (historial de pagos) y previous_application ya que refleja antecedentes enfocados en esta misma institución financiera (no como beruau que se enfoca en otras entidades, esto puede que el modelo encuentre sumamente complicado detectar un patron), es importate destacar que las razones de acptacion o rechazo pueden variar entre instituciones financieras.

### Polars como marco de datos seleccionado
Para este ejercicio en particular debido al volumen de datos suministrado se opta por usar polars en lugar de pandas debido a que optimiza de manera significativa el uso de memoria y los tiempos de computo, por lo que las funciones creadas usan sintaxis declarativa (por ejemplo si bien en pandas puedo acceder a una columna con df['columna_buscada'] con polars seria df.col('columna_buscada'))

### Carga Inicial de Datos

El primer paso para realizar la creacion y evaluacion de un modelo analitico es cargar y entender los datos que nos suministran.

Se cargaron los archivos principales del dataset Home Credit Default Risk, distribuidos en varias tablas relacionadas. Las tablas utilizadas y su descripcion resumida son:

* application_train.csv: Información general del cliente y la variable objetivo (TARGET).

* previous_application.csv: Detalles sobre solicitudes de crédito anteriores.

* installments_payments.csv: Historial de pagos de cuotas.
  
### Analisis preliminar (entendimiento de los datos)

La tabla HomeCredit_columns_description.csv es la mas importante en este analisis preliminar ya que nos permite discriminar el tipo de dato que es cada variable para poder catalogarlas en numericas, fechas, valores binarios o categorias. Tambien permite identificar columnas que no suministran información significativa o que son redundantes.

De este analisis obtenemos que:

* *Variable de union:* La variable `SK_ID_CURR` es el ID unico de cada cliente y esta nos permitira realizar uniones entre tablas.

* *Variable objetivo:* La variable Target es la variable que deseamos predecir en el modelo de aprendizaje supervizado, es de tipo binario en donde
   * 1 → La persona incurrió en mora
   * 0 → La persona no incurrió en mora
* *Columnas que pueden descartarse sin analisis mas detallado:* A continuación se presentan algunas columnas que conviene eliminar por tener poca o ninguna utilidad:
  
| Columna                        | Motivo de descarte                                                                                       |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| `NAME_TYPE_SUITE`             | Despues de un analisis rapido se encuentra que el 80.82% de los registros tienen el valor `'Unaccompanied'`, lo cual indica muy baja diversidad. No aporta segmentación relevante. |
| `DAYS_REGISTRATION`           | Fecha de registro del cliente, no es directamente útil para perfilar comportamiento crediticio.           |
| `DAYS_ID_PUBLISH`             | Fecha de emisión del documento de identidad. Poco valor informativo teniento datos como fecha de nacimiento.                         |
| `WEEKDAY_APPR_PROCESS_START`  | Día de la semana en que se inició la solicitud. No tiene relación con el perfil de riesgo del cliente.     |
| `HOUR_APPR_PROCESS_START` `DAY_APPR_PROCESS_START`    | Hora y día de inicio del proceso. No se considera relevante para clusterización o predicción.                    |
| `APARTMENTS_MEDI` y variables referentes a estadisticos de vivienda   | Estas variables podrian llegar a tener alguna relevancia pero debido al numero de caracteristicas tan grande que engloba el problema es mejor dejar las variables que muestran un omportamiento financiero o que me permitan crear un perfil claro de la persona (para clusterizacion), como por ejemplo genero, edad, etc.                    |
| `FLAG_PHONE`                  | Indica si el cliente proporcionó un número de teléfono fijo. Redundante si ya se considera el móvil.       |
| `FLAG_EMP_PHONE`              | Indica si proporcionó teléfono del trabajo. Redundante si ya se considera el móvil.   |


### Pre-preprocesamiento de la data

Antes de realizar la limpieza del data set es necesario verificar que la data este estructurada y entendible para el modelo ya que por ejemplo el modelo no entiende que significa la M y la F en la columna genero, es necesario cambiarlas a valores numericos o booleanos, voy a llamar a esta etapa el pre-preprocesamiento ya que es el paso anterior a realizar la limpieza de la data.

En esta etapa es necesario:
* Cambiar las variables categoricas por valores numericos donde cada numero representa una categoria, por ejemplo tal y como mencione anteriormente en la categoria genero se reemplaza M por 0 y F por 1.
  para poder proceder a cambiar los valores de las columnas categoricas por valores numericos, es necesario comprobar que este funcionando de manera adecuada la funcion, se va a analizar los valores unicos para cada columna y despues de realizar el cambio de categorias a valores numericos se volvera a evaluar, debe haber la misma cantidad de valores unicos (se adjunta imagen del cambio que hubo en las variables y que se mantiene el numero de categorias).

  Por ejemplo si una categoria es color y tiene valores rojo, verde o azul entonces tiene 3 valores diferentes, al realizar la funcion se pasara a tener valores 0 (rojo), 1 (verde) o 2 (azul) pero seguira         teniendo 3 valores unicos.
  ![image](https://github.com/user-attachments/assets/0b08400c-8c9e-491b-871e-36d18af369ed)

  
* Cambiar las variables numericas de montos y flujos a una escala logaritmica, esto debido a que las transformaciones logarítmicas pueden ayudar a corregir la asimetría de variables y mejorar la linealidad en modelos estadísticos.
  
* Cambiar las unidades de medida en variables que no son comprensibles, por ejemplo en las columnas DAYS_BIRTH o DAYS_EMPLOYED estan medidas en dias negativos por lo que es mucho mas practico pasarlo a meses (se adjunta imagen de como se manejan los datos en estas variables).
  ![image](https://github.com/user-attachments/assets/5011f4df-fba4-4b3c-9849-3d1708360e92)

### Preprocesamiento de la data

Despues de realizar un entendimiento profundo de la data y estructurarla de una manera que pueda manipularla facilmente, ya que se debe eliminar el ruido de los datos para que el modelo no sufra de overfitting, es decir que el modelo considere como validado, solo los datos que se han usado para entrenar el modelo, sin reconocer ningún otro dato que sea un poco diferente a la base de datos inicial [https://protecciondatos-lopd.com/empresas/overfitting/#Que_es_el_overfitting_en_el_aprendizaje_automatico]. 


Para limpiar la data se procede a realizar los siguientes pasos para las tres tablas empleadas en el desarrollo de este trabajo (train, previous_application e installment):

*Para comenzar se combina temporalmente Train y Test en un solo dataframe para realizar el preprocesamiento de manera uniforme

* Manejo de valores faltantes: Se procede a determinar el porcentaje de valores faltantes en cada columna, se va a realizar el conteo de valores faltantes y se expresara como un porcentaje del total, despues se mostrara la distribucion de valores faltantes en las columnas y en base a esta informacion se procedera a definir el Umbral de imputacion (se anexa informacion de la distribucion de valores faltantes, podemos ver que el 75% de los datos (Q3) tiene menos de 50% de valores faltantes por lo que podriamos tomar el 50% como el Umbral de imputacion), despues en las columnas restantes se reemplazaran los valores faltante por la media de dicha columna, para previous_application se pondra un Umbral del 41% y por otro lado se tiene que en installment no se tienen casi valores faltantes.
Para manejar los valores faltantes de las columnas que no se eliminaron, se procede a reemplazar los valores faltantes en las columnas de tipo numerico, para este caso particular voy a proceder a reemplazarlos por la media de la columna.
Para las variables de tipo categorico no tendria sentido reemplazarlos con algun estadistico dado que es una categoria, por ello se procede a eliminar todos los registros que tienen valores faltantes

![image](https://github.com/user-attachments/assets/439af6f2-f788-4a21-bb2e-9cec98afc99f)



* Reestructurar variables para captar alguna dependencia temporal: Despues de revisar en la literatura se encuentra que variables como número de rechazos previos, historial de solicitudes y frecuencia de aceptación son claves para desarrollar un analisis completo. Con esto en mente se busca reinterpretar la columna "NAME_CONTRACT_STATUS" para tener una idea del numero de rechazos y aprobaciones en los ultimos X intentos.

* Reducir caracteristicas: Una tecnica comun en el preprocesamiento de datos es la reduccion de caracteristicas, esta me dice que si tengo 2 vaiables que tienen una correlacion fuerte (si mi correlacion es igual a 1 entonces son directamente proporcionales y si es -1 son inversamente proporcionales) entonces es redundante tenerlas y puede llegar a dificultar la deteccion de un patron ya que lo que me dice una variable se puede interpretar de la otra. Con esto en mente se procede a calcular la correlacion entre las variables y a eliminar las que estan fuertemente correlacionadas entre si.
Se muestra la comparacion del grafico de calor que compara la correlacion de todas las variables numericas evaluadas en este ejercicio, se puede apreciar que despues de realizar la reduccion a caracteristicas principales se pudo disminuir la concentracion de variables altamente relacionadas
![image](https://github.com/user-attachments/assets/f8889885-3b0b-4c31-bb43-9e70b276cfd4)



  
* Deteccion y manejo de valores atipicos: Se debe determinar los valores que se salen del comportamiento normal de la variable, para ello se procedera a ver los valores que son superiores al bigote superior (el bigote superior se calcula como el cuartil 3 + 1.5 * el rango intercuartilico, el rango intercuatilico es la diferencia entre el cuartil 3 y el cuartil 1), y el bigote inferior se calcula como Q1-1.5 * el rango intercuartilico.Si los valores atipicos representan una gran parte de la poblacion entonces la muestra no tiene una comportamiento normal, para esto se determina si los valores atipicos son menores al 7% (este valor no esta respaldado en la literatura, es una asuncion propia) y en caso de que sea menores al 7% se pueden eliminar sin alterar significativamente el resultado.

* Extaer variable objetivo: Se extrae la variable objetivo y se almacena aparte en una matriz Y que contiene el ID del consumidor y si fue aceptado o rechazado (0 o 1)

* Columnas que pueden resumirse: : Hay varios campos que no me dan información significativa de manera individual pero se puede generar un nuevo campo calculado que resuma varias columnas, por ejemplo son 20 columnas que dan informacion sobre si el cliente entrego un documento X, seria mas practico resumir esas 20 columnas en una sola para evaluar cuantos de esos documentos se entregaron con respecto al total de documentos. Ademas podria sacar otra columna como el ingreso por hogar al dividir los ingresos que tiene una persona por el numero de personas que viven en su hogar "CNT_FAM_MEMBERS" y de esta forma elimino "CNT_FAM_MEMBERS".
  
* Por ultimo se separa nuevamente el dataset en conjunto de entrenamiento y prueba y se almacena en un archivo aparte.

# Desarrollo del EDA

Se desarrolla una descripcion grafica de variables de interes para poder entender de manera facil y sencilla como esta compuesta la data, ademas se busca responder a preguntas basicas del comportamiento del dataset.

En el archivo EDA.ipynb esta todo descrito pero de las conclusiones mas interesantes son:
* El dataset esta compuesto principalmente por mujeres lo cual hace que se pueda llegar a pensar que las mujeres incurren mas en mora pero verdaderamente esto es una ilusion generado por su peso en la poblacion.
  ![image](https://github.com/user-attachments/assets/e19bed7d-6385-4f89-b81b-2c3c0c811c39)
  ![image](https://github.com/user-attachments/assets/8e6e955f-adca-43ba-a752-f8b496bb41de)
* Increiblemente la gente con hijos tiende a pagar sus deudas de manera mas responsable que la gente sin hijos, esto puede deberse a que los padres tienen una mayor presion para no aumentar sus deudas debido a mora.
  ![image](https://github.com/user-attachments/assets/2db52429-a51f-4be2-94e5-df4924557c7d)
* Las personas que no incurren en mora tienen una distribucion mucho mas uniforme que quienes si incurren en mora, esto indica que las personas que no incurren en mora pueden tener cualquier tipo de salario pero hay algunos salarios especificos donde se acumulan las personas que incurren en mora, por ejemplo es muy notable que las personas que mas probabilidad tienen de incurrir en mora son las persiben un ingreso neto entre 11.5 y 12 puntos en la escala logaritmica.
  ![image](https://github.com/user-attachments/assets/f5cba5db-6d66-48fb-9c2d-61e23ef491f8)
* Podemos apreciar que tiene una distribucion regular y facil de entender, es claro que mientras menos tiempo lleve una persona trabajando se tiene una mayor concentracion de personas que incurren en mora, esto indica que tener una persona que lleve poco tiempo trabajando es una señal de alerta y un indicativo de que puede incurrir en mora.
![image](https://github.com/user-attachments/assets/9345cd38-d70f-49c3-b635-1e74d8f50598)

### Clusterización por K-means
Para el algoritmo no supervizado voy a optar por usar el algoritmo de k-means donde dividimos la data en k clusters que comparten determinadas caracteristicas, este algoritmo es rapido y escalable sin mencionar que se puede enfocar muy bien al problema, dado que al identificar grupos de clientes con características similares (ej. clientes puntuales vs. morosos frecuentes), se pueden personalizar estrategias comerciales, de riesgo o cobranza.

Se deben seleccionar las variables a utilizar en la clusterizacion de la data, no se deben seleccionar todas las variables debido a que cuando tienes muchas variables en un modelo, especialmente si algunas no son importantes, las diferencias entre las personas se vuelven más difíciles de ver. Imagina que tienes un mapa con muchas calles y caminos, pero muchos de esos caminos no llevan a ningún lugar importante.

Las variables deben abarcar aspectos demográficos, socioeconómicos y financieros para obtener clusters significativos y representativos. K-Means agrupa a las personas en función de la similitud de sus características, por lo que es esencial tener una diversidad de variables que describan el comportamiento financiero desde diferentes ángulos.

Gracias al EDA sabemos que el tiempo que lleva empleado y el ingreso neto afectan significativamente en que una persona incurra en mora, por esto vamos a buscar variables cuantitavas referentes a la los ingresos, deudas, edad, estabilidad laboral y familia.

de esta forma se seleccionan las variables ["CNT_CHILDREN", "REGION_POPULATION_RELATIVE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT", "AMT_APPLICATION", "AMT_GOODS_PRICE_right", "INGRESO_POR_PERSONA", "CONVERTIDO_DAYS_EMPLOYED", "CONVERTIDO_DAYS_BIRTH"]

* Ya hemos preprocesado la data pero para el algoritmo k-means es necesrio realizar un paso adicional y es el escalado de las variables, ya que K-Means es sensible a la escala de las variables. Para este caso se va utilizar StandardScaler de scikit-learn para normalizar los datos.
![image](https://github.com/user-attachments/assets/b2d5448c-f130-47eb-9cc1-e66b2329d1cd)

* Ahora el algoritmo K-means utiliza un numero K de categorias o clusters para etiquetar el comportamiento de la data pero es mi decision cuantos clusters va a tener, voy a utilizar el metodo del codo para determinar el numero optimo de clusters que deberia tener K-means, el metodo del codo se basa en determinar de manera grafica la ganancia que obtengo con forme aumento el numero de clusters (cuando la ganancia tiende a dejar de ser significativa se genera un punto de inflexion en la grafica o un "codo"), el numero donde se genera el codo es el numero optimo de clusters.
  ![image](https://github.com/user-attachments/assets/b1fe6650-a69c-4237-8a1d-1b75a8d8dc21)
  El grafico nos muestra 3 posibles puntos de inflexion que son el 2, el 3 y el 4. Por conveniencia voy a escoger 3 clusters debido a que 2 puede que no recoja toda la informacion requerida y con 4 debo consumir una mayor cantidad de recursos computacionales.

* Para un tema de visualizacion se va a reducir la data a dos componentes principales (es indiferente cuales sean, unicamente es con motivos graficos) y a desglozar por clusters, en la grafica podemos visualizar que no se estan teniendo confuciones y cada categoria engloba una parte de la poblacion que no se confunde con otra.
  ![image](https://github.com/user-attachments/assets/df304023-d3df-49b1-b483-1338d4efd4c9)
*Ya tenemos la clusterizacion de la poblacion pero el modelo K-means separa la poblacion en grupos pero no me dice en ningun momento cuales son los posibles grupos, para ello es necesario desglozar los estadisticos de cada columna segmentado por cluster, se crea un archivo csv que contiene los estadisticos de cada cluster y con este se construye un perfil de cada persona. Del archivo se va a buscar el cuartil 3 que englobe la mayoria de datos que tiene cada cluster, se adjunta imagen de los datos recogidos de variables mas relevantes (presentan cambios notables entre clusters).

![image](https://github.com/user-attachments/assets/9fead4f8-2704-45e8-b1ab-481c2e5459ad)

Con esta informacion describo los perfiles de la siguiente manera:

### Análisis de Clusters

| Cluster | Descripción de la persona |
|---------|----------------------------|
| 0       | Personas mayores que cuentan con un buen ingreso para mantener a todos los miembros de su familia, tienden a pedir créditos de monto medio a largo plazo y tienen un trabajo estable. |
| 1       | Persona adulta con un muy buen ingreso para mantener a todos en su casa, tiene estabilidad laboral y tiende a solicitar montos altos a corto plazo. |
| 2       | Persona joven con ingresos deficientes para mantener a todos en su casa, no tiene mucha estabilidad laboral y tiende a buscar montos bajos a mediano plazo. |

### Desarrollo del modelo de prediccion

Se procede a entrenar un algoritmo con la data preprocesada para que prediga con exita si una persona incurriria en mora o no, para este caso voy a desarrollar el analisis utilizando un random forest (en caso de que no conozcas un random forest es un agrupamiento de arboles de decision que miran un subconjunto de los datos y cada arbol hace una predicción, al final se vota por la clase mas seleccionada).

Primero se exporta la data limpia de entrenamiento, prueba y la variable objetivo (recordemos que estan en files/output/ bajo el nombre de train.csv,test.csv y target.csv respectivamente), se deben extraer unicamente los valores target sin incluir el ID del cliente dado que el modelo espera una variable con la forma (n,) siendo n el numero de registros a evaluar.

Despues de separada la data se procede a entrenar el modelo con unos hiperparametros iniciales que van a mutar despues de cada analisis para poder encontrar los mejores hiperparametros, los hiperparametros relevantes para el random forest son:

* n_estimators: Es el numero de arboles de decision que se encuentran adentro del bosque
* max_depth: Es la profundidad que puede tener cada arbol antes de realizar una predicción
* random_state: No afecta al desempño del modelo, unicamente es un iniciador.

Realice varios entrenamientos variando varios hiper parametros empezando desde (n_estimators=100, max_depth=10)  hasta (n_estimators=400, max_depth=50) evaluando el modelo en cada resultado viendo como variaban sus metricas.

Las metricas que utilice para evaluar el modelo son:

* AUC (Area Under Curve): representa la capacidad del modelo para distinguir correctamente entre clases. Un valor alto de AUC indica que el modelo tiene buena capacidad para identificar correctamente los valores reales y evitar falsos positivos. Un valor cercano a 1 es la mejor metrica y un valor cercano a 0,5 indica que el algoritmo es tan bueno como el azar.
  ![image](https://github.com/user-attachments/assets/8da3bb48-5c85-4c70-9f79-f5fe2b15c7ab)


* Matriz de confusion: muestra cómo se comporta el modelo al clasificar cada clase. Por ejemplo, de todos los casos en los que el valor real era 0, cuántos fueron correctamente clasificados como 0, y de todos los casos en los que el valor real era 1, cuántos fueron correctamente clasificados como 1.
  |               | Predicho: 0 | Predicho: 1 |
|---------------|-------------|-------------|
| Real: 0       | 6922        | 0           |
| Real: 1       | 589         | 0           |

* Variables de peso: Tambien se muestran las variables que tienen mas peso para el modelo a la hora de determinar si una persona entraria en mora o no.
  ![image](https://github.com/user-attachments/assets/58fae107-4672-4b5a-9891-e710d532f61f)


### Analisis

Como se puede apreciar el desempeño del modelo es sumamente pobre ya que su AUC del 0.51 indica que es igual que el azar y la matriz de confusion nos verifica que cataloga a todos como 0 (no entra en mora).
Como habia mencionado antes realice varios entrenamientos modificando los hiperparametros buscando mejorar los resultados pero el comportamiento no mejoraba, el modelo siempre catalogaba a todos como 0. Ahora por que esta pasando eso, estos resultados tienen mucha logica considerando la distribución de los datos en el dataset (volvamos al EDA).
![image](https://github.com/user-attachments/assets/9807c970-94eb-4633-ac01-3594c5ce20c0)

Se puede observar que el 92% de los datos corresponden a personas que no incurren en mora (clase 0). Esto significa que el conjunto de datos está fuertemente desbalanceado, ya que hay una gran diferencia entre la cantidad de ejemplos de la clase 0 (sin mora) y la clase 1 (con mora).

Este desbalance puede hacer que un modelo de clasificación se incline a predecir casi siempre la clase mayoritaria (sin mora), ya que con eso obtendría una alta precisión general, aunque fallaría al identificar correctamente a las personas que sí caen en mora, que es justamente el caso más crítico que se desea detectar.

### Conclusion

En este caso, el modelo de clasificación no resulta útil, ya que está fuertemente sesgado por el desbalance en la distribución de la variable objetivo. La mayoría de los registros pertenecen a clientes que no incurren en mora, lo que provoca que el modelo aprenda a predecir casi exclusivamente esa clase, sin aportar valor en la identificación de los casos relevantes (clientes en mora). Por esta razón, el modelo supervisado no es adecuado para ser implementado en su estado actual.

En cambio, el enfoque no supervisado demostró ser más valioso. Al no depender de la variable objetivo, permitió identificar tres perfiles distintos de clientes basados en sus características. Esta segmentación puede ser utilizada por Home Credit para diseñar portafolios de productos personalizados, adaptados a las necesidades e intereses de cada grupo, mejorando así las oportunidades de conversión y fidelización.


















