import base64
import datetime
import sqlite3
from urllib import request
from flask import request
import common

# db = None


def getBookmarks():
    sql = "SELECT * FROM bookmarks ORDER BY id DESC"

    common.db.row_factory = sqlite3.Row

    cur = common.db.cursor()
    cur.execute(sql)

    # merges keys/values
    rows = dict(result=[dict(r) for r in cur.fetchall()])

    for r in rows['result']:
        r['image'] = base64.b64encode(r['image']).decode()
        r['thumbnail'] = base64.b64encode(r['thumbnail']).decode()
        # r['src_image'] = base64.b64encode(r['src_image'])

    return rows


def deleteBookmark():
    print("DELETE BOOKMARK", request)
    data = request.get_json()
    print("DELETE", data['id'])
    sql = "DELETE FROM bookmarks WHERE id=?"

    cur = common.db.cursor()
    cur.execute(sql, (int(data['id']),))
    common.db.commit()
    return "ok"


def saveBookmark():
    # TODO Include small reference thumbnail
    try:
        data = request.get_json()
        sql = '''
            INSERT INTO bookmarks(
                created, seed, prompt, ddim_steps, width, height, scale, ddim_eta, sampler, strength, thumbnail, image, src_image
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            '''

        image = base64.standard_b64decode(data['image'])
        thumb = base64.standard_b64decode(data['thumbnail'])
        src_image = None
        if 'src_image' in data:
            src_image = base64.standard_b64decode(data['src_image'])

        parms = (
            datetime.datetime.now(),
            int(data['seed']),
            data['prompt'],
            str(data['ddim_steps']),
            str(data['width']),
            str(data['height']),
            str(data['scale']),
            str(data['ddim_eta']),
            data['sampler'],
            data.get('strength', None),
            memoryview(thumb),
            memoryview(image),
            src_image
        )
        cur = common.db.cursor()
        cur.execute(sql, parms)
        common.db.commit()
        return "ok"
    except Exception as e:
        print("EXCEPTION", e)
