# Translation

## AIHub 한국어-영어 번역(병렬) 말뭉치
### - 데이터셋 - 약 160만개의 데이터
### 자원의 문제로 인해 1024*128개로 학습(500Epoch)

## 1. google/mT5-small
```
DatasetDict({
    train: Dataset({
        features: ['SRC', 'TRG'],
        num_rows: 104857
    })
    test: Dataset({
        features: ['SRC', 'TRG'],
        num_rows: 13107
    })
    valid: Dataset({
        features: ['SRC', 'TRG'],
        num_rows: 13108
    })
})
```

- SRC - 영어, TRG - 한국어

Input        : however , professor yoon advised , a variety of ways to achieve qualitative , not quantitative , growth need to be sought for a continuous growth of winter festivals of gangwon do province .  
Prediction   : 다만 윤 교수는 강원도 겨울축제의 지속적인 성장을 위해 다양한 방법이 필요하다고 권고했다 .  
Ground Truth : 다만 윤 교수는 “강원도 겨울축제가 지속해서 성장하려면 양적 성장보다 질적 성장을 위한 다양한 방안을 고민할 필요가 있다"고 조언했다 .   
  
Input        : i pray that our brother and sister , the north korean refugees , will find their way and truth through the lord only , and will receive their lives .  
Prediction   : 북한 난민의 아버지와 남편이 하나님을 통해 길과 진실을 찾고 , 삶을 받을 수 있도록 기원합니다 .  
Ground Truth : 우리 형제자매인 북한이탈주민이 오직 주님을 통해 길과 진리를 찾게 하시며 생명을 얻을 수 있도록 기도합니다 .   

Input        : after taking national road no . from jincheon to cheongju , then exiting at the okseong intersection , there is omi village of okseong ri , munbaek myeon .  
Prediction   : 진천에서 청주까지 국립도로 1호선을 타고 , 삼척면 옹성리 마을이 있다 .  
Ground Truth : 진천에서 국도 17호선을 타고 청주 방향으로 가다가 옥성교차로를 빠져나오면 문백면 옥성리 오미마을이 나온다 .   
  
Input        : in honor of king gwanggaeto s accomplishments , his son and successor , king jangsu , erected a stele the following year at king gwanggaeto s tomb site in the goguryeo capital , gungnaeseong present day ji an , china .  
Prediction   : 장수의 영광을 기념해 , 그의 아버지와 후배가 중국 용만성 선물일지인 고궁지 묘지에서 다음 해 화장실을 세웠다 .  
Ground Truth : 광개토대왕의 업적을 기리고자 ,  그의 아들이자 후계자인 장수왕은 이듬해 고구려 수도인 궁내성(현재 중국 지안)에 위치한 광개토대왕의 묘지에 비석을 세웠다 .   

Input        : as english premier league , cardiff city s emiliano sala went missing due to the plane crash , a recording of his voice at the time of the crash was released .  
Prediction   : 잉글랜드 프리미어리그(UEFA)가 항공기 운전으로 사망한 가운데 카플시티의 에니노 라라가 사고 당시 목소리를 녹음한 것으로 알려졌다 .  
Ground Truth : 잉글랜드 프리미어리그 카디프시티의 에밀리아노 살라(28)가 비행기 추락사고로 실종된 가운데 ,  사고 당시 그의 음성이 담긴 메시지가 공개됐다 .   

## 2. transformer 구조
### 자원의 문제로 인해 1024*128개로 학습(500Epoch)
```
DatasetDict({
    train: Dataset({
        features: ['SRC', 'TRG'],
        num_rows: 104857
    })
    test: Dataset({
        features: ['SRC', 'TRG'],
        num_rows: 13107
    })
    valid: Dataset({
        features: ['SRC', 'TRG'],
        num_rows: 13108
    })
})
```

- SRC - 영어, TRG - 한국어
