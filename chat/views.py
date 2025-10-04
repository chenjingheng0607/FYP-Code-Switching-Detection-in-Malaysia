from django.shortcuts import render

# Create your views here.
def chat_interface(request):
    """
    Renders the chat interface page.
    """
    return render(request, 'chat/chat_interface.html')