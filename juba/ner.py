from requests import put

def Ner(text=''):
    data = {"data": text}  # title非空
    S = put('http://www.pynlp.net/PublicNer/', data=data).json()
    return S['识别结果']