import sqlite3

conn = None


def init():
    global conn

    try:
        conn = sqlite3.connect("db/sd.db", check_same_thread=False)
        print("CONN", conn)
    except Exception as e:
        print("DB EXPCET", e)
