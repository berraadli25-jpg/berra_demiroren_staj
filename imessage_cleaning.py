import os
import re

INPUT_DIR = 'data/raw/chats_new'
OUTPUT_DIR = 'data/raw/chats_parsed'
MERGED_FILE = 'data/raw/imessage_parsed.txt'

SPAM_KEYWORDS = ['text stop to quit', 'you are subscribed', 'order is ready', 'arriving with your', 'track here', 'crave to cut down', "don't share this code"]

def parse_chat_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('\u2028', ' ').replace('\u2029', ' ')
    # remove any line containing "tapbacks"
    content = '\n'.join(line for line in content.split('\n') if 'tapbacks' not in line.lower())

    blocks = re.split(r'\w+ \d+, \d{4}\s+\d+:\d+:\d+ [AP]M[^\n]*\n', content)

    messages = []
    for _, block in enumerate(blocks[1:]):
        text = block.strip()
        if not text:
            continue
        lines = text.split('\n')
        sender = lines[0].strip()
        message = '\n'.join(lines[1:]).strip()
        if not message:
            continue

        # remove attachment paths
        message = re.sub(r'/users/\S+\.(?:png|jpg|jpeg|mov|mp4|pdf|heic)', '', message, flags=re.IGNORECASE).strip()
        # remove standalone image/video filenames
        message = re.sub(r'\b\w+\.(?:png|jpg|jpeg|mov|mp4|heic)\b', '', message, flags=re.IGNORECASE).strip()
        # remove "this message responded to an earlier message." lines
        message = re.sub(r'this message responded to an earlier message\.?', '', message).strip()
        # remove standalone loved/liked/etc by lines
        message = re.sub(r'(loved|liked|disliked|laughed at|emphasized|questioned|reacted) by .+', '', message, flags=re.IGNORECASE).strip()
        # remove urls
        message = re.sub(r'https?\S+', '', message).strip()
        # lowercase
        message = message.lower()

        if not message:
            continue

        # skip spam messages
        if any(kw in message for kw in SPAM_KEYWORDS):
            continue

        prefix = "ME" if sender == "Me" else "THEM"
        messages.append((prefix, message))

    # concatenate consecutive messages from same sender
    merged = []
    for prefix, text in messages:
        if merged and merged[-1][0] == prefix:
            merged[-1] = (prefix, merged[-1][1] + " " + text)
        else:
            merged.append((prefix, text))

    return merged

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"Found {len(files)} chat files")

    all_conversations = []

    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)
        messages = parse_chat_file(filepath)
        if not messages:
            continue
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(f"{prefix}: {text}" for prefix, text in messages))
        all_conversations.append('\n'.join(f"{prefix}: {text}" for prefix, text in messages))

    with open(MERGED_FILE, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_conversations))

    print(f"Saved parsed chats → {OUTPUT_DIR}")
    print(f"Merged all chats → {MERGED_FILE}")
    print(f"\nSample from first file:")
    first = parse_chat_file(os.path.join(INPUT_DIR, files[0]))
    for prefix, text in first[:5]:
        print(f"  {prefix}: {text}")

if __name__ == '__main__':
    main()