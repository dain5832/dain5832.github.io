---
layout: post
title:  "[논문리뷰]WaveNet: A Generative Model for Raw Audio"
author: dain
categories: [paper review]
image: assets/images/12.jpg
featured: true
hidden: true
---

# WaveNet: A Generative Model for Raw Audio

구현여부: No
작성일시: 2021년 1월 12일 오후 9:06

# Abstract

### **WaveNet**

- Raw audio waveform을 생성하는 DNN - Generative Model
- 특징: Fully probabilistic and Autoregressive
    - Fully probabilistic:  random variable과 probability distribution을 사용하여 modeling함.
    - Autoregressive: 변수의 과거 값들을 조합하여 관심있는 변수를 예측함.

⇒ 현재시점 t에서의 audio sample의 predictive distribution을 이전 시점의 audio sample에 대한((Autoregressive)) 조건부 확률로 modeling함(Fully probabilistic).

### **WaveNet의 쓰임**

- Text-to-Speech
- 1개의 model로 여러 화자의 특징을 잡아낼 수 있고, conditioning을 통해 화자에서 화자로 바꿀 수도 있음.

    (이전의 TTS 모델은 대부분 음성 데이터를 쪼개고 조합해서 생성하는 방식이었기 때문에 화자나 톤을 바꾸고자 할 때 마다 새로운 모델/데이터가 필요했음.)

- 음악 생성
- phoneme(음소) recognition

# 1. Introduction

### **Neural Autoregressive Generative Model**

- Generative Model의 일종. Autroregressive를 이용.
- 이미지나 텍스트 등 복잡한 distribution을 modeling한 것.
- 어떻게?

    : Neural Net을 사용하여 픽셀/단어의 joint probability를 modeling 했는데, 이때 joint probability= (조건부 확률의 곱) 임을 이용.

- 이걸 음성(wideband raw audio waveform)에도 적용해보면 어떨까? ⇒ WaveNet의 탄생

### **WaveNet**

- PixelCNN에 기반한 audio generative model
- 음성의 특징인 long-range temporal dependency 문제를 해결하기 위해 Dilated Causal Convolution을 사용함
- 성과
    - TTS 분야에서 자연스로운 raw speech signal을 생성할 수 있음.
    - 하나의 모델로 여러 종류의 목소리를 생성할 수 있음.
    - speech recognition 잘하고, music같은 다른 modality에도 적용가능.

# 2. WaveNet

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled.png)

- 특정 timestep t에서의 $x_t$는 모든 이전 timestep에 대해 condition됨.(조건부 확률)
- 그리고 이 조건부 확률을 모두 곱하면, 특정 timestep t와 그것의 모든 과거시점$(X = \{x_1, x_2, ... , x_T\})$의 joint probability가 된다.

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/1.jpg](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/1.jpg)

참고 1. joint probability를 조건부 확률의 곱으로 나타낼 수 있는 이유 (chain rule) 

- 모델 전반 요약
    - conditional probability distribution을 modeling하기 위해 convolution layer를 쌓음.
    - output과 input의 time dimensionality는 동일.
    - output: softmax를 거져서 나온 categorical distribution(다음 timestep의 값)
    - optimization: loglikelihood를 최대화하는 방향으로 parameter를 update

## 2.1. Dilated Causal Convolutions

- Standard Convolution(참고용)

    ![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%201.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%201.png)

### Causal Convolution

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%202.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%202.png)

- timestep t의 output을 내기 위해 timestep t까지의 input들만을 고려하는 방식
- 목적: CNN을 시계열 데이터에 사용할 때, 모델이 data의 순서를 violate하지 않도록 하기 위함.

    (future timestep이 들어가면 안됨.)

- 과정
    - Training에서는 각 timestep별로 pararell하게 연산 가능.(RNN과의 비교했을 때의 장점)
    - Generation 단계에서는 sequential.
- 장점

    : recurrent connection이 없기 때문에 빠르다 → 특히 data가 길 때 good.

- 단점

    : receptive를 늘리기 위해 많은 수의 레이어 또는 큰 필터를 필요로 한다.

    ⇒ 이 문제를 해결한 것이 Dilated Convolution!

### Dilated Convolution

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%203.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%203.png)

causal conv와 dilated conv가 합쳐진 형태, dilation이 각각 1, 2, 4, 8인 경우

- 특정 step(dilation)마다 input 값을 받는 것.
- 목적: 적은 수의 layer로 넓은 범위의 receptive field를 받는 것 → long term dependency를 capture할 수 있음.
- 장점: 같은 효과를 내는 방법들 중 가장 효율적

    1) 적은 layer수로 large receptive field를 갖게 됨. 연산량 감소 (vs applying large layer)

    2) input shape이 그대로 유지되어 나옴. 정보 유실 적음. (vs stride, pooling)

    3) non-linear 연산이 많이 들어가 model이 더 discriminative해짐.(vs applying large filter)

- WaveNet에서의 활용
    - 512를 limit으로 잡은 뒤, dilation을 layer 마다 2배씩 늘리고, limit에 도달하면 다시 1부터 시작.
        - Intuition 1) Dilation을 2배씩 늘리면 receptive field도 2배씩 증가
        - Intuition 2) block을 stacking하면 model capacity와 receptive field도 더더욱 증가

## 2.2 Softmax Distributions

### Modeling Conditional Distribution

- Softmax Distribution으로 modeling(Classification task)
    - 마지막에 softmax를거쳐서 categorical하게 나옴. (나올 수 있는 모든 값들의 확률을 1-D로 담아서 그 중 확률 제일 큰거 선택)

### Quantization(양자화)

- 무한대의 가지수를 갖는 음성 data의 값을 유한한 몇 가지 대표값으로 바꿔주는 것.

    ex) 0~2 → 0,   2~4 →2,   4~6→4 ...

- Raw audio가 16bit이기 때문에 만약 그대로 쓰게된다면 output으로 가능한 값이 2^16 = 65,536개이고, 이 말은 softmax에서 나타내는 class가 65,536개라는 뜻 → 너무 많다.
- 이를 해결하기 위해 양자화 도입, input data에 mu-law compounding transformation을 사용하여 8-bit(256개)로 바꿔줌 → softmax를 통해 max 값을 정하면 얘를 다시 reconstriction해서 원래 raw audio 값(16bit)으로 바꿔줌.

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%204.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%204.png)

mu-law compounding transformation 공식

- 양자화에는 linear과 nonlinear 방식이 있는데 nonlinear 방식(mu-law compounding)이 더 좋았음

## 2.3 Gated Activation Units

- gated PixelCNN에서 사용된 gated activation과 동일

### 목적

- Pixel CNN이 Pixel RNN에 비해 연산 속도는 빠르지만 성능이 더 낮다는 점을 보완하기 위해 고안됨.

### 수식

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%205.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%205.png)

                       gated activation 수식

- Filter 부분과 Gate 부분의 elementwise mutiplication이라고 볼 수 있음.
- Filter 부분
    - 일반적인 convolution 연산(dilated) 후 activation으로 tanh
    - 특정 layer에서 뽑아진 local feature를 의미.
- Gate 부분
    - 일반적인 convolution 연산 후(dilated) activation으로 sigmoid(0-1사이 값)
    - Intuition: Filter 부분의 값(정보)를 다음 layer에 얼마만큼 정해줄지를 결정함.

### 이게 왜 잘될까?

RNN이 가진 장점을 보완했기 때문이라 생각.

- RNN이 더 잘됐던 원인

    1)  LSTM의 recurrent conntection으로 인해 모든 layer가 이전 pixel의 entire neighborhood에 접근하게 됨. 반면 ReLu를 쓴다면, neighborhood의 영역이 layer 깊이에 따라 linear하게 증가함.

    (WaveNet에서는 해당이 안된다고 생각하는데.. 이해 확인하기)

    2) multiplicative unit이 들어가 더 복잡한 interaction을 model하는데 도움을 주었을 것.

- 내 생각: Swish(Self-Gated Activation)의 형태와 비슷하다고 생각한다.
    - Swish 식 : $f(x) = x\cdot \sigma(x)$
    - ReLU와 달리 negative value에서 미분 값이 0이 되지 않음.(Leaky ReLU 해결) + 그 외 장점

- gated pixelCNN 구조(참고)

    ![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%206.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%206.png)

- Swish 그래프

    ![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%207.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%207.png)

## 2.4 Residual and Skip Connections

### 목적

- Convergence 속도를 높이고 더 깊은 network의 사용을 용이하게 하기 위함.

    (residual connection이 gradient flow에 효과적)

- 1X1 convolution 연산: 연산량을 줄이고, shape을 맞추는 용도

    ![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%208.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%208.png)

    ![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%209.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%209.png)

    Heiga Zen, [https://www.youtube.com/watch?v=nsrSrYtKkT8](https://www.youtube.com/watch?v=nsrSrYtKkT8)

## 2.5 Conditional WaveNets

### 목적

- 원하는 characteristic을 가진 audio를 generate할 수 있게 하기 위함.

### 수식

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2010.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2010.png)

- additional input h를 넣어 conditional distribution을 model하도록 함.

### 방식

- Global Conditioning
    - h가 전체 timestep에서 고정됨 ex) userID(TTS)

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2011.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2011.png)

                                      V는 학습되는 파라미터

- Local Conditioning
    - h가 전체 timestep에서 변함, timeseries로 들어감$(h_t)$ ex) linguistic feature
    - 먼저 input h의 timestep이 audio의 timestep과 같아지게 하기 위해 transposed convolution network를 적용한다. $(y = h(x))$

        →transposed conv를 하는 이유는 보통 h의 sampling frequency가 더 낮기 때문.

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2012.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2012.png)

- 참고: Transposed Convolution

    ![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2013.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2013.png)

## 2.6 Context Stacks

- Receptive field size를 늘릴 수 있는 또 다른 방법.

### 방법

- 두 개(또는 N개)의 WaveNet을 stack하는 것.

    → receptive field가 작은 shorter-range WaveNet(large)와 receptive field가 넓은 longer-range WaveNet(small)

- Shoter-range WaveNet이 메인 모델이고, 여기에 local conditioning을 하자! 이때 추가로 들어올 input $h_t$는 longer range WaveNet의 output(softmax안함).

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/context_stack.jpg](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/context_stack.jpg)

### 효과

- Longer-range WaveNet의 output이 input으로 들어오기 때문에 Receptive field는 longer range WaveNet의 receptive field만큼이 됨.
- Longer-range WaveNet에서는 pooling layer가 사용될 수 있기 때문에 같은 receptive field를 가진 single WaveNet과 비교했을 때 연산량을 줄일 수 있음.

*두 WaveNet이 모두 전체 data를 보는 것인지, 아니면 나눠서 일부만 보는 것인지 햇갈리는데 답변해주시면 감사하겠습니다!

# 3. Experiments

## 3.1. Multi-Speaker Speech Generation

- Dataset: English multi-speaker corpus from CSTR voice cloning toolkit
- 위에서 나온 Global conditioning을 활용하여 User ID를 새로운 input h로 넣어 학습 진행(one-hot vector 형태).
- User ID 를 함께 넣어주면 이후 generation 단계에서 해당 화자의 목소리로 output을 냄.
- training 단계에서 해당 화자의 목소리만 가지고 training하는 것보다 다른 화자들의 목소리도 함께 사용해서 training 했을 때 성능이 더 좋았음. → wavenet의 internal representation이 multiple speaker들 사이에서 share된다.
- voice 뿐 아니라 다른 음성적 특징(음질, 숨소리 등)도 함께 잡아냄.

## 3.2 Text-to-Speech

- Dataset: North American English and Mandarine Chinese dataset
- Local conditioning을 활용하여 각각 linguistic feature와 $F_0$값(=기본주파수)에 condition시켜 학습 진행.
- 기존 모델들보다 성능 좋았고, WaveNet끼리 비교했을 때는 linguistic feature에만 condition되었을 때보다 linguistic feature와 $F_0$값 모두에 condition되었을 때의 성능이 더 좋았음.

## 3.3 Music

- Dataset: MagnaTagATune, YouTube piano dataset
- Tag를 Global Conditioning을 통해 추가 input으로 넣었으면(binary vector이용), tag에 맞는 음악이 생성됨.

## 3.4 Speech Recognition

- Dataset: TIMIT
- Generation이 아닌 discrimination task(맞추는거)
- Network 구조 살짝 변형
    - dilated convolution 사이에 [mean pooling layer→ non-causal convolution] 넣음
    - Loss : 다음 sample predict용, classification용
- TIMIT data에서 best score냄.

# 4. Conclusion

### WaveNet

- Deep generative model for audio data(waveform)
- 특징: Autoregressive, dilated convolution
- 다른 input에 global 또는 local하게 condition됨으로서 원하는 특징을 가진 output을 만들어낼 수 있음.
- TTS, music audio modeling, speech recognition task에서 good.

# Appendix

## A. Text-to-Speech Background

- Goal : text를 speech로 render하는 것(sequence to sequence mapping problem)
- 사람은 이를 어떻게 하는가? → 이를 비스무리하게 컴퓨터에 적용해보자
- TTS pipeline

    1) Text analysis part: input = word sequence, output = phoneme sequence

    2) Speech synthesis part: input = phoneme sequence, output = speech waveform

- Speech synthesis(pipeline의 두번째 부분)에서의 2가지 Main approach

    1) non-parametric, concatenative approach

    2) statistical parametric approach

- Statistical parametric approach의 과정

![WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2014.png](WaveNet%20A%20Generative%20Model%20for%20Raw%20Audio%20cc7bc522253f428da17e7beb3283f65b/Untitled%2014.png)

등등등

## B. Details of TTS Experiment

- sampling과 실험에 사용된 data setting
- 사용된 linguistic feature
- subjective listening test가 어떤 방식으로 이루어졌는지

### 출처)

mu-law compounding [https://m.blog.naver.com/PostView.nhn?blogId=sbkim24&logNo=10084099777&proxyReferer=https:%2F%2Fwww.google.com%2F](https://m.blog.naver.com/PostView.nhn?blogId=sbkim24&logNo=10084099777&proxyReferer=https:%2F%2Fwww.google.com%2F)

Gated PixelCNN [https://arxiv.org/pdf/1606.05328.pdf](https://arxiv.org/pdf/1606.05328.pdf)

Swish [https://medium.com/techspace-usict/swish-a-self-gated-activation-function-3b7e551dacb5](https://medium.com/techspace-usict/swish-a-self-gated-activation-function-3b7e551dacb5)

Generative Model-Based Text-to-Speech Synthesis [https://www.youtube.com/watch?v=nsrSrYtKkT8](https://www.youtube.com/watch?v=nsrSrYtKkT8) 

[논문리뷰] WaveNet [https://joungheekim.github.io/2020/09/17/paper-review/](https://joungheekim.github.io/2020/09/17/paper-review/)

모두의 연구소 김성일님 발제영상 [https://www.youtube.com/watch?v=GyQnex_DK2k](https://www.youtube.com/watch?v=GyQnex_DK2k)

Generative Model-Based Text-to-Speech Synthesis [https://www.youtube.com/watch?v=nsrSrYtKkT8](https://www.youtube.com/watch?v=nsrSrYtKkT8)

GIthub-issue: context stack부분 관련 [https://github.com/ibab/tensorflow-wavenet/issues/164](https://github.com/ibab/tensorflow-wavenet/issues/164)

long-term dependency 관련 [https://brunch.co.kr/@chris-song/9#:~:text=장기 의존성](https://brunch.co.kr/@chris-song/9#:~:text=%EC%9E%A5%EA%B8%B0%20%EC%9D%98%EC%A1%B4%EC%84%B1)(Long%2DTerm%20Dependency,%EB%8F%84%EC%9B%80%EC%9D%84%20%EC%A4%84%20%EC%88%98%20%EC%9E%88%EC%A3%A0.

Transposed Convolution [https://dataplay.tistory.com/29](https://dataplay.tistory.com/29)