import requests
import json


def run():
    data = {
        'age': '20',
        'name': 'sam',
        'time': '2:1121232',
        'img_url': 'www.naver.com'
    }

    EndPoint = "http://192.168.0.12:5000/face_recog"

    res = requests.post(url=EndPoint, data=json.dumps(data))

    print(res)


if __name__ == '__main__':
    run()