from django.http import JsonResponse
# from .utils.chatbot_helpers import get_bot_reply
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from langchain_core.messages import HumanMessage

from .forms import ChatForm
from .services.chat_services_streaming import chatbot_graph


def index(request):
    form = ChatForm()
    return render(request, 'chatbot/index.html', {'form': form})


# def chat_api(request):
#     print(request)
#     if request.method == "POST":
#         form = ChatForm(request.POST)
#         if form.is_valid():
#             user_message = form.cleaned_data['message']
#             bot_reply = get_bot_reply(user_message)
#             return JsonResponse({'reply': bot_reply})
#         else:
#             return JsonResponse({'error': 'Invalid input'}, status=400)
#
#     return JsonResponse({'error': 'Invalid method'}, status=405)


# streaming
@csrf_exempt
def chat_stream(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")

        def event_stream():
            # stream_mode="messages" ensures each step comes separately
            for message_chunk, metadata in chatbot_graph.stream(
                    {'messages': [HumanMessage(content=user_message)]},
                    config={"configurable": {"thread_id": "1"}},
                    stream_mode="messages",
            ):
                if message_chunk.content:
                    # send each step as SSE chunk
                    yield f"{message_chunk.content}\n\n"

        return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
