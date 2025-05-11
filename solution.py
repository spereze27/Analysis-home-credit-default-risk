import pandas as pd

# Carga de archivos principales
app_train = pd.read_csv("files/application_train.csv")
"""
bureau = pd.read_csv("files/bureau.csv")
bureau_balance = pd.read_csv("files/bureau_balance.csv")
previous = pd.read_csv("files/previous_application.csv")
pos_cash = pd.read_csv("files/POS_CASH_balance.csv")
installments = pd.read_csv("files/installments_payments.csv")
credit_card = pd.read_csv("files/credit_card_balance.csv")
columns_desc = pd.read_csv("files/HomeCredit_columns_description.csv", encoding='latin-1')

columns_desc[columns_desc['Table'] == 'application'][['Row', 'Description']]
print(columns_desc)
"""


import pandas as pd

# Cargar la tabla
app_train = pd.read_csv("files/application_train.csv")

# Total de filas (incluyendo nulos)
total = len(app_train)

# Filas donde NAME_TYPE_SUITE es 'Unaccompanied'
unaccompanied_count = (app_train['NAME_TYPE_SUITE'] == 'Unaccompanied').sum()

# Porcentaje
percentage = 100 * unaccompanied_count / total

print(f"Clientes 'Unaccompanied': {unaccompanied_count} ({percentage:.2f}%) del total")





