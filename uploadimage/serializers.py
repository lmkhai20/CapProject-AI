from rest_framework import serializers
from .models import UploadModel


class UploadSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(required=True)

    class Meta:
        model = UploadModel
        fields = ['image']
