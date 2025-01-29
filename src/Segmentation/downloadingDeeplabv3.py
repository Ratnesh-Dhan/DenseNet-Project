# -*- coding: utf-8 -*-
# DOWNLOADING deeplabv3 FROM KAGGLE

import kagglehub

try:
    # Download latest version
    path = kagglehub.model_download("tensorflow/deeplabv3/tfLite/metadata", force_download= True)
    
    print("path to model files: ", path)
except Exception as e:
    print("Error: ",e)
    
# path =  C:\Users\NDT Lab\.cache\kagglehub\models\tensorflow\deeplabv3\tfLite\metadata\2
# this is the download folder i guess