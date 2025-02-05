import json
from datetime import datetime
import re

no_of_messages = 37
sign = "⊙"
pattern =fr"{re.escape(sign)}.*?{re.escape(sign)}" 
false_users=[]
def recode(s):
    return s.encode('latin1').decode('utf-8')
def write(messages, out, out_messages, with_date=False):
    for message in messages:
        if "content" in message.keys():
            name = recode(message['sender_name'])
            content = recode(message['content'])
            s = sign + name + sign + ': ' + content.replace('\r\n', ' ').replace('\n', ' ')
            if with_date:
                timestamp = message["timestamp_ms"]
                date_time = datetime.fromtimestamp(timestamp//1000)
                formatted_date = date_time.strftime('%Y-%m-%d %H:%M:%S')
                s = formatted_date + ' ' + s
            out_messages.write(s + '\n')
            json.dump({"text": s}, out, ensure_ascii=False)
            out.write('\n')
            matches = re.findall(pattern, content)
            for match in matches:
                false_users.append(match)

messages = []
for i in range(1,no_of_messages):
    f = open(f"messages/message_{i}.json", "r")
    j = json.load(f)
    messages += j["messages"]
    
messages.sort( key= lambda x : x['timestamp_ms'])
out_clear = open("messages.jsonl", "w", encoding="utf-8")
out_messages = open("messages.txt", "w", encoding="utf-8")
write(messages, out_clear, out_messages)
if len(false_users) != 0:
    raise BaseException("sign exist in text!")
out_clear.close()
out_messages.close()