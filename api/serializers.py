from rest_framework import serializers
from .models import CsvFilePath

class CsvFileSerializer(serializers.Serializer):
    file = serializers.FileField()


class CsvFilePathSerializer(serializers.ModelSerializer):
    class Meta:
        model = CsvFilePath
        fields = '__all__'

