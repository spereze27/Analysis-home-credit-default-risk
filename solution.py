import pandas as pd

# Carga de archivos principales
app_train = pd.read_csv("files/application_train.csv")
"""bureau = pd.read_csv("/bureau.csv")
bureau_balance = pd.read_csv("/bureau_balance.csv")
previous = pd.read_csv("/previous_application.csv")
pos_cash = pd.read_csv("/POS_CASH_balance.csv")
installments = pd.read_csv("/installments_payments.csv")
credit_card = pd.read_csv("/credit_card_balance.csv")
#columns_desc = pd.read_csv("HomeCredit_columns_description.csv", encoding='latin-1')
"""
print(app_train.head())
