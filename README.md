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















