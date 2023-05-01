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


class TrainModels(APIView):

    def post(self,request):
        model = request.data
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
