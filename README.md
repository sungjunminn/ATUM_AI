# AI_FlaskServer
---------------

## Project 목적
Atum Studio의 Rule-Designer서비스에서 ChatBot 서비스를 제공하기 위해 Machine Learning 알고리즘이 돌아가는데, Atum Studio는 Spring Boot framework을 사용하여 Python 코드를 안정적으로 돌리기 어렵습니다.
그래서 이를 Flask Server를 사용하여 API로 만들었습니다. 

----------------
## 실행 방법
`python app.py`

---------------
## 사용하는 Python Package (최근 업데이트: 22/09/29)

한번에 설치하는 방법: 프로젝트를 내려받은 후 
`python setup.py install`

flask==2.2.2
flask-Cors==3.0.10
pandas==1.4.4
numpy==1.23.3
konlpy==0.6.0
sklearn==0.0
PyMySQL==1.0.2
keras==2.10.0
tensorflow==2.10.0

---------------
## 추가 설치 사항

ImportError: DLL load failed while importing _jpype: 지정된 모듈을 찾을 수 없습니다. 발생 시
> visual c++ redistributable for visual studio 2015 설치 ( CPU 비트수에 맞게 설치 )

---------------

