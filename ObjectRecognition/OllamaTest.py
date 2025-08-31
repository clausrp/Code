import os
import json
from typing import TypedDict


from dotenv import load_dotenv
from imap_tools import MailBox, AND
 
from langchain.chat_models import init_chat_model
from langchain_core.tools import tools

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END


load_dotenv

IMAP_HOST = os.getenv('IMAP_HOST')
IMAP_USER = os.getenv('IMAP_USER')
IMAP_PASSWORD = os.getenv('IMAP_PASSWORD')
IMAP_FOLDER = 'INBOX'


CHAT_MODEL = 'qwen3:8b'


class ChatState(TypedDict):
    messages: list

def connect():
    mail_box = MailBox(IMAP_HOST)
    mail_box.login('IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER')
                   
    return mail_box

def list_unread_emails():
    """Return a bullet list of every UNREAD message's UID, subject, data and sender"""

    print('List unread emails tool called')

    with connect() as mb:
        unread = list(mb.fetch(criteria= AND(seen=False), headers_only=True, mark_seen=False))

    if not unread:
        return 'you have no unreads'
    
    response = json.dumps([
        {
            'uid': mail.uid,
            'date': mail.date.astimezone().strftime('%Y-%m-%d %H:%M'),
            'subject': mail.subject,
            'sender': mail.from_
        } for mail in unread

    ])