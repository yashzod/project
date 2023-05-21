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
from django.shortcuts import redirect
from .ml_algo import linear_regression
# Create your views here.

class UploadFileView(APIView):

    def get(self, request):
        data = request.body.decode("utf-8")
        print(data,"******************************************")
        return Response({'msg':'uploaded'},status=status.HTTP_200_OK)


    def post(self,request):
        print("*******************************************",request.data)

        serializer = CsvFileSerializer(data={'file':request.data['file']})
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",request.data['file'])
        if serializer.is_valid():
            path =  default_storage.save('file.csv',request.data['file'])
            print(path)
            serializer = CsvFilePathSerializer(data={'path':path})
            if serializer.is_valid():
                serializer.save()
            df = pd.read_csv(io.StringIO(default_storage.open(path).read().decode('utf-8')))
            columns = df.columns
            # return redirect('https://s6e2u5.csb.app/')
            return Response({'msg':'uploaded','path':path,'columns':columns},status=status.HTTP_200_OK)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',serializer.error_messages)
        return Response({'msg':'not uploaded','error':serializer.error_messages},status=status.HTTP_400_BAD_REQUEST)

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
        print(request.data)
        data = [
                [
                    {
                    "x": [1, 2, 3, 4, 5],
                    "y1": [1, 3, 2, 4, 5],
                    "y2": [2, 4, 3, 5, 6],
                    "y3": [3, 5, 4, 6, 7]
                    },
                    {
                    "x": [1, 2, 3, 4, 5],
                    "y1": [5, 4, 3, 2, 1],
                    "y2": [6, 5, 4, 3, 2],
                    "y3": [7, 6, 5, 4, 3]
                    }
                ],
                [
                    {
                    "x": [1, 2, 3, 4, 5],
                    "y1": [10, 9, 8, 7, 6],
                    "y2": [11, 10, 9, 8, 7],
                    "y3": [12, 11, 10, 9, 8]
                    },
                    {
                    "x": [1, 2, 3, 4, 5],
                    "y1": [6, 7, 8, 9, 10],
                    "y2": [7, 8, 9, 10, 11],
                    "y3": [8, 9, 10, 11, 12]
                    }
                ]
                ]
        # return Response({"data":data},status=status.HTTP_200_OK)

        model = request.data['model']
        # train_split = request.data['train_test_split']['train']
        # test_split = request.data['train_test_split']['test']
        # validation_split = request.data['train_test_split']['validation']
        file_name = request.data['file_name']
        df = pd.read_csv(io.StringIO(default_storage.open(file_name).read().decode('utf-8')))
        # column_processing = request.data['column_processing']
        # processed_df = process_df(df,column_processing)
        x_cols = ["a",'b','c','d']
        y_col = "e"
        data = linear_regression(df,train_test_split=request.data['train_test_split'],x_cols=x_cols,y_col=y_col)
        print(data)


        return Response({"data":data},status=status.HTTP_200_OK)


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

