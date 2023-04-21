import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import BytesIO
from skimage.io import imread
from skimage.transform import resize
import streamlit as st
import pickle
from PIL import Image
import random
import datetime
from azure.cosmos import exceptions, CosmosClient

st.set_option('deprecation.showfileUploaderEncoding',False)

st.title('Intelligence Storage For Image classification')
st.subheader('Unlocking the power of computer vision through image classification')
model=pickle.load(open('img.pkl','rb'))
upload_file=st.file_uploader('Choose an image',type='jpg')
if upload_file is not None:
  img=Image.open(upload_file)
  imgName = img
  st.image(img,caption='Image uploaded')

if st.button('Predict'):
  labels=['Flower','Ball','Ice- Cream']
  st.write('Result')
  flat_data=[]
  img=np.array(img)
  img_resized=resize(img,(150,150,3))
  flat_data.append(img_resized.flatten())
  flat_data=np.array(flat_data)
  y_out=model.predict(flat_data)
  y_out=labels[y_out[0]]
  st.title(f'Predicted output: {y_out}')
  q=model.predict_proba(flat_data)
  for index,item in enumerate(labels):
    st.write(f'{item} : {round(q[0][index]*100, 4)}%')

  from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Create a BlobServiceClient object using the connection string
  connection_string = "DefaultEndpointsProtocol=https;AccountName=imgclass;AccountKey=2KjJktKesG01ula/N7tVrNcJajJPhrMSoS04KTiTmjjJCh4fQkhascLpiDxX1AsvGItvrN1ESutt+ASta+kJAw==;EndpointSuffix=core.windows.net"
  blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create a container client for your new container
  container_name = "imagestorage"
  container_client = blob_service_client.get_container_client(container_name)

# Upload a file to the container
  blob_name = y_out + "_" + str(random.randint(1, 100))
  blob_client = container_client.get_blob_client(blob_name)
  with BytesIO() as output:
        imgName.save(output, format="JPEG")
        data = output.getvalue()
        blob_client.upload_blob(data, overwrite=True)

  # Download the test file from the blob
  downloaded_blob = blob_client.download_blob()
  downloaded_data = downloaded_blob.content_as_bytes()

# Get the existing metadata
  metadata = imgName.info

# Add the timestamp to the metadata
  timestamp = datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S')
  metadata['DateTime'] = timestamp

  # Define your Cosmos DB account information
  url = 'https://imagedata.documents.azure.com:443/'
  key = 'WZv2z7T2HU6Ii93VrCipfKi7ifldJFGG5a7Ixcdp3ALGGoXimkrPzrxr9GsfWfUpbd1YhMGbOkFLACDbqlUYXw=='
  database_name = 'imgdata'
  container_name = 'container1'

# Create a Cosmos DB client
  client = CosmosClient(url, credential=key)

# Get a reference to the database and container
  database = client.get_database_client(database_name)
  container = database.get_container_client(container_name)

  item = {
    'id':blob_name,
    'jfif': metadata.get('jfif'),
    'jfif_version': metadata.get('jfif_version'),
    'jfif_unit': metadata.get('jfif_unit'),
    'jfif_density': metadata.get('jfif_density'),
    'progression': metadata.get('progression'),
    'progressive': metadata.get('progressive'),
    'DateTime': metadata.get('DateTime'),

    }

# Insert the item into the container
  response = container.create_item(body=item)



st.text('Cloud Computing Project of Group-1 D12C')
st.text('Members : Shubham, Muskan, Mansi & Khushi')
