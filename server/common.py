import sqlite3

model = None
modelCS = None
modelFS = None

db = None


def init():
    global db

    try:
        db = sqlite3.connect("db/sd.db", check_same_thread=False)
        print("DB", db)
    except Exception as e:
        print("DB EXCEPT", e)
