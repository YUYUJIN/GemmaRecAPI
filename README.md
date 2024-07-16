# GemmaRecAPI
> 선호도 기반 유저들의 설문작성 내용을 토대로 PostgreSQL로 구성된 Database를 server 환경에 구축하고 활용.
> 사용자 데이터를 기반으로 Custom 2-stage-model(LRURec 모델과 LLM 모델) 사용해 사용자 히스토리 기반 추천 아이템 제안 서비스를 제공하는 API.  
> LRURec 모델을 통해 사용자 히스토리를 기반으로 후보군을 검색하고, 사용자 세부정보/히스토리/생성한 후보군으로 prompt를 작성하여 LLM 모델의 입력으로 사용하여 Verbalizer 레이어를 사용하여 추천 아이템 반환.  
> FastAPI 형태로 선호 기반 설문 내용을 Database 형태로 관리하고, 이를 활용해 한 번의 API 호출로 train 및 inference를 진행하는 것을 목표로 작성.  
> 구축된 Database를 기반으로 학습 데이터들을 최신화하며, 추가적인 user group, item 들이 추가 되어도 추천 가능한 LLM 기반 추천 시스템.

<img src=https://img.shields.io/badge/python-3.10.0-green></img>
<img src=https://img.shields.io/badge/transformer-4.42.3-yellow></img>
<img src=https://img.shields.io/badge/Flask-3.0.3-blue></img>
<img src=https://img.shields.io/badge/postgreSQL-2.9.9-orange></img>
<img src=https://img.shields.io/badge/pytorch-2.3.1-red></img>  

## How to use
 Recurrent, attention 기반 LRURec 모델과 LLM 모델(본 프로젝트에서는 GPU 자원 제한으로 Huggingface의 google/gemma-2b 모델 사용)을 사용하기 위해 pytorch등 기반 dependency를 설치한다.  
 환경에 맞게 설치하되, 본 프로젝트의 개발 환경인 Linux에 맞는 설치는 다음과 같다.
```
pip3 install torch torchvision torchaudio
pip install psycopg2-binary, transformer
```  
  
 사전준비 및 필요 dependency 설치   
```
git clone https://github.com/YUYUJIN/GemmaRecAPI
cd GemmaRecAPI
pip install -r requirements.txt
```

 workplace 내에 .env 파일을 만들어 Database의 정보생성
```
HOST={host ip}
PORT={port number(postgresql default:5432)}
USERNAME={user name}
PASSWARD={user password}
DATABASENAME={database name}
HFTOKEN={huggingface token}
```
 본 프로젝트에서는 server 내에 Database를 이용하였고, 엔진으로는 PostgreSQL로 구성하였다.
 (사용 데이터 및 Database의 자세한 내용은 dataBaseSet 폴더 참조)

 사용한 데이터의 mapping, args, 모델 가중치를 saveFiles 폴더에 넣어 프로젝트 폴더 내로 옮긴다.  
 혹은 API 구동 후 /train를 호출하여 모델 학습 및 저장을 진행한다.  

 이후 app을 구동하여 간이 API를 사용한다.  
```
python app.py
```
## Structure
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/structure.png></img>  
프로젝트의 구조도는 위와 같다.  
  
 LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking 논문에서 제공한 구조와 구현을 대부분 따르고, 이를 위해 설문 데이터를 논문에 사용된 데이터와 비슷한 구조를 가지게 변형하여 사용하였다.  
 제시된 논문에서처럼 배포된 사전학습 4bit 양자화 LLM 모델을 LoRA 기반으로 파인튜닝 진행하여 사용하였다. 사용한 모델은 Gemma-2B 모델이고, 파인튜닝 시에는 AWS의 g5.xlarge 인스턴스 환경에서 확보한 데이터 기준 6시간 정도 소요되었다.  
 학습에 사용한 Prompt는 PALR: Personalization Aware LLMs for Recommendation 논문에서 제안하는 Prompt 양식을 참고하여 작성하였다. 후술할 LLM(Gemma-2B) 내용 참조.  

## LRURec && LLM(Gemma-2B)
LRURec 모델에 사용한 데이터 예시와 Test 데이터셋에 대한 모델의 성능평가.    
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/LRUdata.JPG></img>  
<학습 데이터 예시>  
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/LRUresult.JPG></img>  
<Test 데이터셋 기준 성능평가>
  
 LLM 모델 Fine-tuning, Inference 시 사용한 Prompt 및 Test 데이터셋에 대한 모델의 성능평가  
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/LLMprompt.JPG></img>  
<Prompt 예시>  
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/LLMresult.JPG></img>  
<Test 데이터셋 기준 성능평가> 

## API Document
 프로젝트의 결과로 제공할 수 있는 API는 다음 명세와 같다.  
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/api1.JPG></img>  
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/api2.JPG></img>  
 API Document link: https://documenter.getpostman.com/view/15695216/2sA3kPp4Yi 

## Trouble Shooting
<details>
<summary>데이터셋의 효용성</summary>
 확보한 데이터셋을 논문에 사용한 데이터셋 형태로 변형하여 사용하면 타겟 uid(유저) 당 매칭되는 sid(아이템)의 표본이 적어 누락되거나, 학습이 잘 일어나지 않는 경우가 있다. 자세한 해결방안은 dataBaseSet 내용 참조. 
</details>

<details>
<summary>LLM 모델 교체</summary>
 기반 논문에서 제시한 LLama2-7B 모델에서 Gemma-2B로 교체하여 구성하였다. 가장 큰 이유로는 서버 GPU 환경에 영향이 제일 크다. 4bit 양자화와 LoRA 방식을 사용하여 학습 파라미터를 약 1.8% 수준으로 줄여도 batch size 4 기준으로 제공 서버에서 파인튜닝을 진행하기에 빠듯하였다.  
 모델 교체 시에 LLama2 구현체와 논문의 구현체, Gemma 구현체를 참조하여 모델의 구현체를 생성하였다. transformer 라이브러리에 구현된 Gemma 모델을 기반으로 논문의 구현체의 임베딩 부분을 참조하여 구성하였다. 논문은 LLama2 기반의 모델으로, Gemma 구현체와 cache, embading에 주의를 두고 구현하였다. 
</details>

<details>
<summary>학습 속도 안정화</summary>
 초기 개인 서버 환경(3070Ti)에서 학습 속도가 불안정해지는 상황이 발생하였다. 학습 데이터 로드 시에 메인스레드만을 이용하는 것과 GPU memory의 사용량 문제로 확인하였다. 이에 학습 시 데이터 로드를 메인 스레드가 아닌 코어 4개를 사용하여 학습하도록 구성하고, evaluate 및 특정 iteration마다 속도를 확인하여 cuda 환경의 여유분이 되는 메모리를 확보하기 위해 cache 메모리를 지우도록 콜백함수를 사용하였다.
 서버 환경을 AWS g5.xlarge 인스턴스로 변경한 이후에는 추가적인 작업 없이도 학습속도는 안정화 되었다. 
</details>

## Reference
 PALR: Personalization Aware LLMs for Recommendation : https://arxiv.org/pdf/2305.07622

 LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking : https://arxiv.org/abs/2311.02089

 github : https://github.com/Yueeeeeeee/LlamaRec

## Produced by Yujin
 <img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/logo.png style="width:100px; height:100px;"></img>

