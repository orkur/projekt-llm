from pathlib import Path

file_path = Path("messages.txt")

def get_users(messages):
    dict = {}
    for message in messages:
        name = message.split(":", 1)[0]
        dict[name + ":"] = 1
    return list(dict.keys())



if file_path.is_file():
    with file_path.open("r", encoding="utf-8") as file:
        content = file.read().splitlines()
        users = get_users(content)
        with open("users.txt", "w", encoding="utf-8") as users_file:
            users_file.write("\n".join(users))
            users_file.close()
else:
    raise FileNotFoundError(f"File {file_path} don't exist!") 