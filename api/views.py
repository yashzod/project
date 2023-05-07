from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from .serializers import CsvFileSerializer, CsvFilePathSerializer
import pandas as pd
import io
from time import sleep
from .helper import process_df
# Create your views here.

class UploadFileView(APIView):

    def post(self,request):
        serializer = CsvFileSerializer(data={'file':request.data['file']})
        if serializer.is_valid():
            path =  default_storage.save('file.csv',request.data['file'])

            serializer = CsvFilePathSerializer(data={'path':path})
            if serializer.is_valid():
                serializer.save()
            df = pd.read_csv(io.StringIO(default_storage.open(path).read().decode('utf-8')))
            columns = df.columns
            return Response({'msg':'uploaded','path':path,'columns':columns},status=status.HTTP_200_OK)
        return Response({'msg':'not uploaded','error':serializer.error_messages},status=status.HTTP_404_NOT_FOUND)

class AttributesView(APIView):

    def post(self, request):
        print(request.data,"***************************")
        model = request.data['model']
        file_name = request.data['file_name']
        df = pd.read_csv(io.StringIO(default_storage.open(file_name).read().decode('utf-8')))
        columns = df.columns
        fields = []
        for col in columns:
            missing_value = {
                "label": col+' missing value',
                "name": col+'_missing_value',
                "type": "select",
                "options":[
                    {"label":'forward fill','value':'forward_fill'},
                    {"label":'backward fill','value':'backward_fill'},
                    {"label":'interpolate','value':'interpolate'},
                ]
                }
            fields.append(missing_value)

            encoding = {
                "label": col+' encoding',
                "name": col+'_encoding',
                "type": "select",
                "options":[
                    {"label":'One Hot','value':'one_hot'},
                    {"label":'Dummy','value':'dummy'},
                    {"label":'Effect','value':'effect'},
                    {"label":'Binary','value':'binary'},
                    {"label":'BaseN','value':'base_n'},
                    {"label":'Hash','value':'hash'},
                    {"label":'Target','value':'target'},
                ]
                }
            fields.append(encoding)

            feature_scaling = {
                "label": col+' feature_scaling',
                "name": col+'_feature_scaling',
                "type": "select",
                "options":[
                    {"label":'Min Max','value':'min_max'},
                    {"label":'Normalization','value':'normalization'},
                    {"label":'Standardization','value':'standardization'},
                ]
                }
            fields.append(feature_scaling)

        fields.append(
            {"label": "k",
                "name": "k",
                "type": "select",
                "options": [
                    {
                    "label": "one",
                    "value": "1"
                    },]}
        )
        
        return Response({'fields':fields},status=status.HTTP_200_OK)
        

class TrainModels(APIView):

    def post(self,request):
        model = request.data['model']
        train_split = request.data['train_test_split']['train']
        test_split = request.data['train_test_split']['test']
        validation_split = request.data['train_test_split']['validation']
        file_name = request.data['file_name']
        df = pd.read_csv(io.StringIO(default_storage.open(file_name).read().decode('utf-8')))
        column_processing = request.data['column_processing']
        processed_df = process_df(df,column_processing)
        print(df)
        testt = request.data['column_processing'][0]['column_name1']['missing_value']

        print(model,file_name,validation_split,testt)
        return Response({'msg':'success'},status=status.HTTP_200_OK)


{
    "model":"linearregression",
    "file_name":"file_name",
    "train_test_split":{
        "train":70,
        "test":20,
        "validation":10
    },
    "column_processing":[
        {
            "column_name1":{
                "missing_value":"average",
                "encoding":"one_hot",
                "feature_scaling":"0-1"
            },
            "column_name2":{
                "missing_value":"average",
                "encoding":"one_hot",
                "feature_scaling":"-1-1"
            }
        }
    ],
    "validation_type":"K-fold"
}
