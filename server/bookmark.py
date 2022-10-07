import base64
import datetime
import hashlib
import io
import sqlite3
from urllib.request import urlopen

from flask import request
from PIL import Image
from rich import print

import common
from common import console


def imgStrToThumb(img: str):
    thumb_bytes = io.BytesIO()
    i = Image.open(io.BytesIO(img))
    i = i.resize([64, 64], resample=Image.LANCZOS)
    i.save(thumb_bytes, format="jpeg", quality=95)
    return base64.b64encode(thumb_bytes.getvalue()).decode()


def imgToHash(image: Image):
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()


def hasMediaHash(hash):
    cur = common.db.cursor()
    res = cur.execute("SELECT COUNT(*) FROM media WHERE hash=?", [hash])

    return (res.fetchone()[0] > 0)


def saveMedia(img, is_src_image=False) -> str:
    decoded = base64.b64decode(img)
    src_image = Image.open(io.BytesIO(decoded))

    hash = imgToHash(src_image)

    if hasMediaHash(hash):
        return hash

    # save media
    cur = common.db.cursor()
    cur.execute("INSERT INTO media(hash, blob, is_src_image) VALUES (?, ?, ?)",
                [hash, decoded, is_src_image])

    common.db.commit()

    return hash


def saveDataURLMedia(img_dataurl, is_src_image=False) -> str:
    with urlopen(img_dataurl) as imgdata:
        src_image_data = imgdata.read()
        src_image = Image.open(io.BytesIO(src_image_data))
        hash = imgToHash(src_image)

        if hasMediaHash(hash):
            # already have it, just pass back the hash
            return hash

        # save media
        cur = common.db.cursor()
        cur.execute("INSERT INTO media(hash, blob, is_src_image) VALUES (?, ?, ?)",
                    [hash, src_image_data, is_src_image])

        common.db.commit()

        return hash


def getBookmarks():
    sql = "SELECT * FROM bookmarks ORDER BY id DESC"

    common.db.row_factory = sqlite3.Row

    cur = common.db.cursor()
    cur.execute(sql)

    # merges keys/values
    rows = dict(result=[dict(r) for r in cur.fetchall()])

    for r in rows['result']:
        if 'image' in r:
            r['thumbnail'] = imgStrToThumb(r['image'])
            r['image'] = None

        if 'src_image' in r:
            r['src_image'] = base64.b64encode(r['src_image'])

    return rows


def deleteBookmark():
    console.log("DELETE BOOKMARK", request)
    data = request.get_json()
    sql = "DELETE FROM bookmarks WHERE id=?"

    cur = common.db.cursor()
    cur.execute(sql, (int(data['id']),))
    common.db.commit()
    return "ok"


def saveBookmark():
    data = request.get_json()
    sql = '''
        INSERT INTO bookmarks(
            created, seed, prompt, ddim_steps, width, height, scale, ddim_eta, sampler, strength, src_hash, img_hash
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        '''

    img_hash = None
    src_hash = None

    print(data.keys())

    if 'image' in data:
        print("Saving image")
        img_hash = saveMedia(data['image'])
    if 'src_image' in data:
        print("Saving src_image")
        src_hash = saveDataURLMedia(data['src_image'], is_src_image=True)

    # TODO: If we have img already, skip saving; there's zero chance an identical hash is a different prompt/etc.

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
        str(data.get('strength', None)),
        src_hash,
        img_hash
    )
    cur = common.db.cursor()
    cur.execute(sql, parms)
    common.db.commit()
    return "ok"
