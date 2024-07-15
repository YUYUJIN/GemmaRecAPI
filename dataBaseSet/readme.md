# GemmaRecDatabase
> 사용자 설문 기반 추천 시스템 개발을 위한 데이터 전처리 및 데이터베이스 구축  

## Train Data
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/aihub.jpg></img>  
link: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71446  
데이터로는 AIHub의 연도별 패션 선호도 파악 및 추천 데이터를 사용. 

## Inforamtion
 설문 내용 총 247729개의 데이터에서 설문 대상자 5215명, 대상 아이템 77510개의 데이터를 사용.  
 1. 설문 내용 중 사용자-아이템 기반 질문 - 선호도, 스타일 관련 질문 8개 채택 후 데이터베이스화(Image 내용의 surveygroup 이미지 참조)  
 2. 설문 내용 중 사용자 관련 질문 - 성별, 직업, 결혼유무, 스타일 관련 질문 6개 채택 후 데이터베이스화(Image 내용의 usergroup 이미지 참조)  
 3. 설문 내용 중 아이템 관련 코멘트 - 이미지이름, 스타일 분류 채택 후 데이터베이스화(Image 내용의 item 이미지 참조)  

## Fixes
 아이템 기준 설문 응답이 6개 이하인 경우 제외, 응답 유저 기준 응답한 아이템이 6개 이하인 경우 제외.  
 추가적으로 응답 유저 기준 아이템이 논문에서 사용한 데이터 대비 낮은 개수를 가져, 후보군 검출 모델인 LRU 모델에서 많이 낮은 성능을 보임(Test 데이터셋 기준 recall@10 0.04)  
 응답 유저를 채택한 특성 기준 총 768개의 그룹으로 분리하여 응답자 그룹핑 진행, 결과로 458 그룹이 생성됨(특성 기준으로 가능한 그룹이 768개, 최종적으로 데이터에서 존재하는 그룹은 458 그룹).

## Image
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/surveygroup.jpg></img>  
<surbeygroup 테이블>  
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/usergroup.jpg></img>  
<usergroup 테이블>  
<img src=https://github.com/YUYUJIN/GemmaRecAPI/blob/main/pictures/items.jpg></img>  
<items 테이블>  

## Reference
연도별 패션 선호도 파악 및 추천 데이터: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71446 