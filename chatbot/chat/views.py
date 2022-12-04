from django.shortcuts import render
from multiprocessing import context
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.template.loader import get_template


# Create your views here.
from django.http import HttpResponse


def home(request):
    context = {}
    return render(request,'chats/index.html',context)