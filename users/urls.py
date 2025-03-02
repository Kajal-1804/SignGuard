from django.urls import path
from . import views

urlpatterns = [
    path('homepage/', views.home, name='homepage'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('verify/', views.verify_signature, name='verify_signature'),
    
]
