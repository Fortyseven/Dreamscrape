import io
from urllib import response
from urllib.request import urlopen

from flask import request
from PIL import Image
from rich import print

import common


def getImage():
    if not 'hash' in request.args:
        raise Exception("No hash provided")

    hash = request.args['hash']
    cur = common.db.cursor()
    res = cur.execute("SELECT blob FROM media WHERE hash=?", [hash])
    img = res.fetchone()[0]
    if not 'thumb' in request.args:
        return (img, 200, {
            'content-type': 'image/png'
        })

    else:
        width: float = 64

        if 'width' in request.args:
            width = int(request.args['width'])

        thumb_bytes = io.BytesIO()
        i = Image.open(io.BytesIO(img))

        nh = int(width * (i.height/i.width))
        i = i.resize([width, nh],
                     resample=Image.LANCZOS)
        i.save(thumb_bytes, format="jpeg", quality=95)

        return (
            thumb_bytes.getvalue(),
            200,
            {'content-type': 'image/png'}
        )
