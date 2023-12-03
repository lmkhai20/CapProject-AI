from django.db import models


def upload_to(instance, filename):
    return 'images/{filename}'.format(filename=filename)


class UploadModel(models.Model):
    image = models.ImageField(upload_to=upload_to)

