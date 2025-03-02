from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):  # Extends Django's built-in authentication
    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')])
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.username
