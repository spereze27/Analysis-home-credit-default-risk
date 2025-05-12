### Analisis de riesgo de credito 
A continuacion se presenta el desarrollo de un modelo de predicción de capacidad de pago de creditos de los clientes basado en diferentes datos suministrados por Home Credit en Kaggle.
Para este ejercicio se trabajara sobre un subconjunto de datos que contiene los datos de la tabla train/test (tabla que contiene toda la informacion general), installments_payments que contiene informacion del comportamiento financiero (historial de pagos) y previous_application ya que refleja antecedentes enfocados en esta misma institución financiera (no como beruau que se enfoca en otras entidades, esto puede que el modelo encuentre sumamente complicado detectar un patron), es importate destacar que las razones de acptacion o rechazo pueden variar entre instituciones financieras.

### Carga Inicial de Datos

El primer paso para realizar la creacion y evaluacion de modelos analiticos utiles para este ejercicio es cargar y entender los datos que nos suministran.
Se cargaron los archivos principales del dataset Home Credit Default Risk, distribuidos en varias tablas relacionadas. Las tablas y su descripcion resumida son:

* application_train.csv: Información general del cliente y la variable objetivo (TARGET).

* bureau.csv y bureau_balance.csv: Historial crediticio del cliente con otras entidades.

* previous_application.csv: Detalles sobre solicitudes de crédito anteriores.

* POS_CASH_balance.csv: Comportamiento de crédito en punto de venta.

* installments_payments.csv: Historial de pagos de cuotas.

* credit_card_balance.csv: Actividad de tarjetas de crédito.

* HomeCredit_columns_description.csv: Diccionario con la descripción de cada variable.
  
### Analisis preliminar (entendimiento de los datos)

La tabla HomeCredit_columns_description.csv es la ms importante en este analisis preliminar ya que nos permite discriminar el tipo de dato que es cada variable para poder catalogarlas en numericas, fechas, valores binarios o categorias. Tambien permite identificar columnas que no suministran información significativa o que son redundantes.

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
| `HOUR_APPR_PROCESS_START`     | Hora de inicio del proceso. No se considera relevante para clusterización o predicción.                    |
| `FLAG_PHONE`                  | Indica si el cliente proporcionó un número de teléfono fijo. Redundante si ya se considera el móvil.       |
| `FLAG_EMP_PHONE`              | Indica si proporcionó teléfono del trabajo. Redundante si ya se considera el móvil.   |

* *Columnas que pueden resumirse:* : Hay varios campos que no me dan información significativa de manera individual pero se puede generar un nuevo campo calculado que resuma varias columnas, por ejemplo son 20 columnas que dan informacion sobre si el cliente entrego un documento X, seria mas practico resumir esas 20 columnas en una sola para evaluar cuantos de esos documentos se entregaron con respecto al total de documentos

### Pre-preprocesamiento de la data
Antes de realizar la limpieza del data set es necesario verificar que la data este estructurada y entendible para el modelo ya que por ejemplo el modelo no entiende que significa la M y la F en la columna genero, es necesario cambiarlas a valores numericos o booleanos, voy a llamar a esta etapa el pre-preprocesamiento ya que es el paso anterior a realizar la limpieza de la data.

En esta etapa es necesario:
* Cambiar las variables categoricas por valores numericos donde cada numero representa una categoria, por ejemplo tal y como mencione anteriormente en la categoria genero se reemplaza M por 0 y F por 1
* Cambiar las variables numericas de montos y flujos a variables logaritmicas, esto debido a que las transformaciones logarítmicas pueden ayudar a corregir la asimetría de variables y mejorar la linealidad en modelos estadísticos.
* Cambiar las unidades de medida en variables que no son comprensibles, por ejemplo en las columnas DAYS_BIRTH o DAYS_EMPLOYED estan medidas en dias negativos por lo que es mucho mas practico pasarlo a meses
* Agrupar las categorias en numericas o categoricas para extraer los estadisticos relevantes para un analisis posterior. Por ejemplo en las variables numericas se puede calcular media, maximo, minimo y desviacion estandar, por otro lado en las variables categoricas  lo mas relevante ess conocer la moda y el numero de valores unicos que tiene la categoria

### Preprocesamiento de la data

Despues de realizar un entendimiento profundo de la data y estructurarla de una manera que pueda manipularla facilmente, ya que se debe eliminar el ruido de los datos para que el modelo no sufra de overfitting, es decir que el modelo considere como validado, solo los datos que se han usado para entrenar el modelo, sin reconocer ningún otro dato que sea un poco diferente a la base de datos inicial [https://protecciondatos-lopd.com/empresas/overfitting/#Que_es_el_overfitting_en_el_aprendizaje_automatico]. 














