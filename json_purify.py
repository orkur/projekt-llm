import json
from datetime import datetime

def recode(s):
    return s.encode('latin1').decode('utf-8')
def write(messages, out): 
    for message in messages:
        if "content" in message.keys():
            timestamp = message["timestamp_ms"]
            date_time = datetime.fromtimestamp(timestamp//1000)
            formatted_date = date_time.strftime('%Y-%m-%d %H:%M:%S')
            name = recode(message['sender_name'])
            content = recode(message['content'])
            out.write(formatted_date + ' [' + name + ']: ' + content.replace('\n', ' ')+"\n" )

messages = []
for i in range(1,37):
    print(i)
    f = open(f"messages/message_{i}.json", "r")
    j = json.load(f)
    messages += j["messages"]
    
messages.sort( key= lambda x : x['timestamp_ms'])
out_json = open("messages.json", "w")
json.dump(messages, out_json, indent=4)
out_clear = open("messages.txt", "w")
write(messages, out_clear)
