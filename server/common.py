import sqlite3
import rich
from rich import console

model = None
modelCS = None
modelFS = None

db = None

console = console.Console()


def init():
    global db, console

    try:
        db = sqlite3.connect("db/sd.db", check_same_thread=False)

    except Exception as e:
        print("DB EXCEPT", e)
