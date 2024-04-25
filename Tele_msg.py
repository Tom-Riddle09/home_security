import telepot



# Replace 'YOUR_API_TOKEN' with your actual Telegram Bot API token
API_TOKEN = '6857629973:AAGcLilQOjuQdilWsUX-EeiyFqu5ZQhlY60'
bot = telepot.Bot(API_TOKEN)

'''def get_chat_id():
    url = f"https://api.telegram.org/bot{API_TOKEN}/getUpdates"
    response = requests.get(url)
    data = response.json()
    if data['ok']:
        # Extract chat ID from the response
        for i in range(len(data['result'])):
            chat_id = data['result'][i]['message']['chat']['id']
            name = data['result'][i]['message']['chat']['first_name']
            username = data['result'][i]['message']['chat']['username']
            print(f'Chat Id: {chat_id}\nName:{name}\nUsername: {username}\n')
    else:
        print("Failed to fetch chat ID.")
        return None'''
    
def send_msg(img_path):
    chat_id = 1153564506
    # Send the photo along with the message
    try:
        bot.sendMessage(chat_id, text="INTRUDER ALERT!")
        bot.sendPhoto(chat_id, photo=open(img_path, "rb"), caption="Captured Image of Intruder!")
        print("Message and image sent successfully!")
    except Exception as e:
        print(f"Error sending message: {e}")

'''if __name__ == "__main__":
    a = int(input('select one \n1.get user chat id:\n2.test send messages\n>'))
    if a == 1:
        get_chat_id()
    elif a == 2:
        chat_id = int(input("enter the chat-id:"))
        print("Sending message to the chat user..")
        asyncio.run(send_msg(chat_id))'''
#def run():
    #asyncio.run(send_msg())

# chat id : Adarsh: 1153564506
    #frinnah : 5315773285