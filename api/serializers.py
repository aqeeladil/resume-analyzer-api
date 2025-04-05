from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "password"]
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user


class AnalysisResponseSerializer(serializers.Serializer):
    analysis = serializers.CharField()
    matches = serializers.ListField(child=serializers.CharField())
    match_percentage = serializers.FloatField()
    missing_skills = serializers.ListField(child=serializers.CharField())
    notice = serializers.CharField(required=False)