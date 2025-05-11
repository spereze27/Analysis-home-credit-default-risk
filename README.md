# Analisis de riesgo de credito 
A continuacion se presenta el desarrollo de un modelo de predicción de capacidad de pago de creditos de los clientes basado en diferentes datos suministrados por Home Credit en Kaggle.

# Carga Inicial de Datos

El primer paso para realizar la creacion y evaluacion de modelos analiticos utiles para este ejercicio es cargar y entender los datos que nos suministran.
Se cargaron los archivos principales del dataset Home Credit Default Risk, distribuidos en varias tablas relacionadas. Las tablas y su descripcion resumida son:

* application_train.csv: Información general del cliente y la variable objetivo (TARGET).

* bureau.csv y bureau_balance.csv: Historial crediticio del cliente con otras entidades.

* previous_application.csv: Detalles sobre solicitudes de crédito anteriores.

* POS_CASH_balance.csv: Comportamiento de crédito en punto de venta.

* installments_payments.csv: Historial de pagos de cuotas.

* credit_card_balance.csv: Actividad de tarjetas de crédito.

* HomeCredit_columns_description.csv: Diccionario con la descripción de cada variable.
  
# Analisis preliminar (entendimiento de los datos)

La tabla HomeCredit_columns_description.csv es la ms importante en este analisis preliminar ya que nos permite discriminar el tipo de dato que es cada variable para poder catalogarlas en numericas, fechas, valores binarios o categorias. Tambien permite identificar columnas que no suministran información significativa o que son redundantes.

De este analisis obtenemos que:

* *Variable objetivo:* La variable Target es la variable que deseamos predecir en el modelo de aprendizaje supervizado, es de tipo binario en donde
   * 1 ->a que la persona incurrio en mora
   * 








