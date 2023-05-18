#!bin/sh

import json
import pandas as pd

data_dir='/home/cristian/Escritorio/Tesis/Codigos_Profesor/'
file = open(data_dir+ 'annotationsTACO.json')
raw_data=json.load(file)

data=pd.DataFrame(raw_data['annotations'])
images=pd.DataFrame(raw_data['images'])
cats=pd.DataFrame(raw_data['categories'])

enum=enumerate(cats['supercategory'].unique())
category_dictionary=dict((j,i) for i,j in enum)

data=pd.merge(data,images,left_on='image_id',right_on='id')
data=pd.merge(data,cats,left_on='category_id',right_on='id')
#scaled_bbox = lambda w,h,bbx : [bbx[0]/w,bbx[1]/h,(bbx[0]+bbx[2])/w,(bbx[1]+bbx[3]/h)]

scaled_bbox = lambda w,h,bbx : [bbx[0]/w,bbx[1]/h,(bbx[2])/w,(bbx[3])/h]
data['scaled_bbox']=data.apply(lambda row : scaled_bbox(row.width,row.height,row.bbox),axis=1)
data[['file_name','supercategory','scaled_bbox']].to_csv('annotations.csv',header=False,index=False)

for image_id in data['image_id'].unique().tolist():
    single_image=data.loc[data.image_id==image_id]
    print('{}'.format(image_id),end="\t")
    print('{}'.format(4),end="\t")
    print('{}'.format(5),end="\t")
    print('{}'.format(single_image['width'].iloc[0]),end="\t")
    print('{}'.format(single_image['height'].iloc[0]),end="\t")
    for (category,bbox) in zip(single_image['supercategory'],single_image['scaled_bbox']):
        print('{}'.format(category_dictionary[category]),end="\t")
        print('{}'.format(bbox[0]),end="\t")
        print('{}'.format(bbox[1]),end="\t")
        print('{}'.format(bbox[2]),end="\t")
        print('{}'.format(bbox[3]),end="\t")
    print('{}'.format(single_image['file_name'].iloc[0]),end="\n")
