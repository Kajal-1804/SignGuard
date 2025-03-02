from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('homepage/', include('users.urls')),
    path('register/', include('users.urls')),  # Register route
    path('login/', include('users.urls')),     # Login route
    path('verify/',include('users.urls')),       # Upload route
    path('', lambda request: redirect('homepage/')),  # Redirect root to register

    path('', include('users.urls')),  # Include users app URLs
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
