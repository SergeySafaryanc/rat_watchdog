# import requests
# import vk_api
# import random
#
# def write_msg(user_id, vk,  message):
#     vk.method('messages.send', {'user_id': user_id, 'message': message, 'random_id': random.randint(0, 2048)})
#
# def send_message(message):
#     token = "bb6cff0b111d5b04c2a1269a41475d3e72f8ccb2f573b9c62fc42d41499e50acf03887dcfa442aba4fcc8"
#     vk = vk_api.VkApi(token=token)
#     subscribers = [57701130, 113761714]
#     for i in subscribers:
#         write_msg(i, vk, message)