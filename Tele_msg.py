import telepot



# Replace 'YOUR_API_TOKEN' with your actual Telegram Bot API token
API_TOKEN = 'YOUR API TOKEN HERE'
bot = telepot.Bot(API_TOKEN)

    
def send_msg(img_path):
    chat_id = #CHAT ID HERE
    # Send the photo along with the message
    try:
        bot.sendMessage(chat_id, text="INTRUDER ALERT!")
        bot.sendPhoto(chat_id, photo=open(img_path, "rb"), caption="Captured Image of Intruder!")
        print("Message and image sent successfully!")
    except Exception as e:
        print(f"Error sending message: {e}")
