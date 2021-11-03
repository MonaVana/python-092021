import os
import pandas
import numpy
import psycopg2
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect

#Dreviny:
HOST = "czechitaspsql.postgres.database.azure.com"
PORT = 5432
USER = "monikavanzurova"
USERNAME = f"{USER}@czechitaspsql"
DATABASE = "postgres"
PASSWORD = "KzR0Q23Q3E0loXUD"

engine = create_engine(f"postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}", echo=False)
inspector = inspect(engine)
#print(inspector.get_table_names())

df = pandas.read_sql(f"dreviny", con=engine)

#1:
smrk = pandas.read_sql("SELECT * from \"dreviny\" WHERE dd_txt = 'Smrk, jedle, douglaska'", con=engine)
nahodila_tezba = pandas.read_sql("SELECT * FROM \"dreviny\" WHERE druhtez_txt = 'Nahodilá těžba dřeva'", con=engine)

#2:
smrk.sort_values(by=["rok"]).plot(x="rok", y="hodnota", kind="bar", title="Vývoj objemu těžby")
#plt.show()

#3:
nahodila_tezba_pivot = pandas.pivot_table(nahodila_tezba, index="rok", columns="prictez_txt", values="hodnota", aggfunc=numpy.sum)
nahodila_tezba_pivot.plot.bar(stacked=True)
#plt.show()

#doplněk:
#kůrovcová kalamita + orkán Kyrill v 2007

#Chicago Crime:
crime = pandas.read_sql(f"crime", con=engine)
#1:
crime_vehicle = pandas.read_sql("SELECT * FROM crime WHERE \"PRIMARY_DESCRIPTION\" = 'MOTOR VEHICLE THEFT'", con=engine)
#2:
crime_vehicle_filtered = crime_vehicle[crime_vehicle["SECONDARY_DESCRIPTION"] == "AUTOMOBILE"]
#3:
crime_vehicle_filtered["date"] = pandas.to_datetime(crime_vehicle_filtered["DATE_OF_OCCURRENCE"])
crime_vehicle_filtered["month"] = crime_vehicle_filtered["date"].dt.month
top_month = crime_vehicle_filtered["month"].mode()
print(top_month)
#září