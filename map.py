from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import pandas as pd
import plotly_express as px

from bson.json_util import dumps

uri = "mongodb+srv://ragi:q1w2e3r4.!@cluster0.tys9hdc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["licenseToSteal"]
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

locC = db["location"]
locations = dumps(list(locC.find()))

df = pd.read_json(locations)
print(df)
fig = px.density_mapbox(df, lat='lat', lon='long', z='val', radius=20, center=dict(lat=df.lat.mean(), lon=df.long.mean()), zoom=4, mapbox_style='open-street-map', height=900)
fig.show()