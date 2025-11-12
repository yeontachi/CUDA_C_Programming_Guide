# Programming_Model(2)

---

## Memory Hierarchy
GPU에서 각 스레드는 실행 중에 여러 종류의 메모리에 접근할 수 있다.
![alt text](Images/image.png)

첫째, **스레드 전용 메모리** 가 있다. 각 스레드는 자신만 사용할 수 있는 **레지스터** 와 **로컬 메모리** 를 가진다. 레지스터는 가장 빠른 저장 공간이며, 로컬 메모리는 전역 메모리 위에 잡히지만 스레드에만 보이는 공간이다.

둘째, **블록 단위의 공유 메모리(shared memory)** 가 있다. 같은 블록에 속한 모든 스레드는 이 메모리를 함께 사용하며, 블록이 종료되면 함께 소멸한다. 공유 메모리는 전역 메모리에 비해 훨씬 빠른 접근 속도를 제공하고, 블록 내 스레드간 협력에 필수적이다. 최근에는 여러 블록을 묶은 **클러스터(thread block cluster)** 단위에서도 블록 간 공유 메모리에 대한 읽기$\cdot$쓰기$\cdot$원자적(atomic) 연산이 가능하도록 확장되었다.

셋째, **모든 스레드가 접근 가능한 전역 메모리(global memory)** 가 있다. 전역 메모리는 GPU 전체에 접근할 수 있으며, 용량이 크지만 지연(latency)이 길고 대역폭 제약이 있다.

추가로, 읽기 전용 메모리 공간도 있다. **상수 메모리(constant memory)**와 **텍스처 메모리(texture memory)**가 그 예이다. 상수 메모리는 용량은 작지만(64KB), 모든 스레드가 동일한 데이터를 빠르게 읽을 수 있도록 최적화되어 있다. 텍스처 메모리는 특정 데이터 형식에 대해 **주소 변환 모드**와 **데이터 필터링** 기능을 제공하여 영상 처리와 같은 작업에서 유용하다.

마지막으로, 전역$\cdot$상수$\cdot$텍스처 메모리는 **커널 실행이 끝나더라도 유지되며,** 동일한 애플리케이션 내의 여러 커널 호출에서 계속 사용할 수 있다는 점이 특징이다.


## Heterogeneous Programming
CUDA 프로그래밍 모델은 **호스트(host)** 와 **디바이스(device)** 라는 두 개의 실행 주체를 전제로 한다. 일반적으로 호스트는 CPU에서 C++ 프로그램을 실행하고, 디바이스는 GPU에서 CUDA 스레드(커널)를 실행하는 보조 프로세서(coprocessor) 역할을 한다.

이 구조의 특징은 **메모리 공간이 분리** 되어 있다는 점이다. CPU는 **호스트 메모리(Host memory)** 를, GPU는 **디바이스 메모리(Device memory)** 를 각각 유지한다. 따라서 전역 메모리(global memory), 텍스처 메모리(texture memory)와 같은 디바이스 측 메모리는 CUDA 런타임 API를 통해 관리해야 한다. 구체적으로는 디바이스 메모리의 **할당 및 해제**, 그리고 **호스트와 디바이스 간 데이터 전송** 을 프로그래머가 명시적으로 처리해야 한다.

그러나 CUDA는 이러한 불편함을 줄이기 위해 **Unified Memory(통합 메모리)** 를 제공한다.
통합 메모리는 CPU와 GPU 모두 **공통된 주소 공간** 을 공유하도록 만들어 주며, 프로그래머가 데이터를 별도로 복사하지 않고도 접근할 수 있도록 한다. 이를 통해 디바이스 메모리를 초과하는 데이터(Oversubscription)도 자동으로 관리할 수 있고, 기존 CPU 프로그램을 GPU 환경으로 이식(porting)하는 작업이 훨씬 단순해 진다.

## Asynchronous SIMT Programming Model
CUDA 프로그래밍 모델에서 계산이나 메모리 연산의 최소 단위는 **스레드(thread)**이다. 전통적으로 스레드는 SIMT(Single Instruction, Multiple Threads) 방식으로 동작하며, 연산과 메모리 접근이 동기적으로 진행된다. 그러나 **NVIDIA Ampere 아키텍처** 부터는 **비동기 프로그래밍 모델**이 도입되어, 메모리 연산의 성능을 크게 가속할 수 있게 되었다.

비동기 프로그래밍 모델은 크게 두 가지 핵심 요소를 정의한다.

**1. 비동기 베리어(Asynchronouss Barrier)** 
 CUDA 스레드 간의 동기화를 담당하는 새로운 메커니즘이다.
 기존의 `__syncthreads()`는 모든 스레드가 지정된 지점에 도달해야 다음으로 진행할 수 있었지만, 비동기 베리어는 **메모리 연산과 계산을 겹치도록(overlap)** 허용하여 성능을 개선한다.
 즉, 스레드들은 데이터 전송이 완료되기를 기다리지 않고, 가능한 연산을 먼저 수행할 수 있다.

**2. 비동기 메모리 복사(cuda::memcpy_async)**
 전역 메모리(global memory)에서 데이터를 비동기적으로 가져올 수 있는 기능을 제공한다.
 이때 스레드들은 데이터가 복사되는 동안에도 **다른 계산을 동시에 수행** 할 수 있다.
 이를 통해 GPU는 **데이터 이동과 연산을 겹쳐 실행(pipelining)** 하여, 메모리 지연(latency)로 인한 성능 저하를 줄인다.

### Asynchronous Operations
비동기 연산은 **CUDA 스레드가 시작하지만 실제 실행은 다른 스레드가 수행하는 것처럼(as-if)** 동작하는 연산을 의미한다.
즉, 어떤 스레드가 연산을 요청하면, 그 연산은 독립적으로 진행되며 요청한 스레드가 반드시 그 결과를 기다릴 필요는 없다.

잘 구성된 프로그램에서는 하나 이상의 CUDA 스레드가 이러한 비동기 연산의 완료 시점과 동기화하게 된다. 여기서 중요한 점은 **연산을 시작한 스레드가 반드시 동기화 과정에 참여할 필요는 없다** 는 것이다. 연산을 시작한 스레드와는 별개로, 동기화는 다른 스레드 집합이 수행할 수 있다.

CUDA에서는 이러한 비동기 연산을 위한 **가상의 스레드(as-if thread)** 라는 개념을 둔다. 이 가상의 스레드는 항상 연산을 요청한 실제 CUDA 스레드와 연결되어 있으며, 실제로는 **동기화 객체(synchronization object)** 를 통해 연산의 완료 여부를 관리한다.

이 동기화 객체는 두 가지 방식으로 관리될 수 있다.

**1. 사용자가 직접 관리하는 경우** : 예: `cuda::memcpy_async`
**2. 라이브러리 내부에서 암묵적으로 관리하는 경우** : 예: `cooperative_groups::memcpy_async`

비동기 연산에 사용되는 동기화 객체로는 대표적으로 `cuda::barrier` 와 `cuda::pipeline` 이 있다.
 - `cuda::barrier` 는 스레드 간의 동기화 지점을 제공하여, 특정 연산이 완료될 때까지 여러 스레드가 함께 대기할 수 있도록 한다.
 - `cuda::pipeline` 은 데이터 이동과 계산을 겹쳐 실행(pipelining)할 수 있게 하여, 메모리 지연을 줄이는 데 효과적이다.

마지막으로, 이러한 동기화 객체는 **다양한 스레드 스코프(scope)** 에서 사용할 수 있다. 스코프란, 어떤 스레드 집합이 특정 동기화 객체를 공유하며 동기화할 수 있는지를 정의하는 개념이다. 
예를 들어,
 - **thread scope** : 한 스레드 내에서만 동기화
 - **warp scope** : 같은 워프 내 스레드들 간 동기화
 - **block scope** : 같은 블록 내 모든 스레드 간 동기화
 - **cluster scope** : 같은 클러스터에 속한 여러 블록 간 동기화
 - **grid scope** : 전체 그리드 차원에서의 동기화

![alt text](<Images/image copy.png>)

## Compute Capability
CUDA 프로그래밍 모델에서 **Compute Capability(연산 능력)** 은 GPU 디바이스가 지원하는 하드웨어 기능을 나타내는 버전 번호이다. 이를 **SM version**이라고도 부르며, 실행 중인 애플리케이션은 이 버전 정보를 통해 GPU가 어떤 기능과 명령어를 지원하는지 확인할 수 있다.

Compute Capability는 **주(major) 리비전 번호 x와 부(minor) 리비전 번호 Y** 로 구성되며, 보통 `X.Y` 형태로 표시된다.

 - **주 번호(major number)** 는 GPU의 핵심 아키텍쳐를 의미한다.같은 주 번호를 가진 GPU들은 동일한 기본 아키텍처를 공유한다.
 - **부 번호(minor number)** 는 같은 아키텍쳐 내에서 점진적인 개선이나 새로운 기능의 추가를 반영한다.

아래 표는 주요 GPU 아키텍처와 그에 대응하는 리비전 번호를 정리한 것이다.

**GPU Architecture and Major Revision Numbers**

| Major Revision Number | NVIDIA GPU Architecture     |
|------------------------|-----------------------------|
| 9                      | NVIDIA Hopper GPU Architecture |
| 8                      | NVIDIA Ampere GPU Architecture |
| 7                      | NVIDIA Volta GPU Architecture  |
| 6                      | NVIDIA Pascal GPU Architecture |
| 5                      | NVIDIA Maxwell GPU Architecture|
| 3                      | NVIDIA Kepler GPU Architecture |


**Incremental Updates in GPU Architectures**

| Compute Capability | NVIDIA GPU Architecture   | Based On                   |
|--------------------|---------------------------|-----------------------------|
| 7.5                | NVIDIA Turing GPU Architecture | NVIDIA Volta GPU Architecture |

여기서 주의할 점은, Compute Capability 버전과 CUDA 플랫폼 버전을 혼동하면 안 된다는 것이다.
 - 예를 들어, *CUDA 7.5, CUDA 8, CUDA 9* 등은 CUDA 소프트웨어 플랫폼의 버전을 의미한다.
 - 반면, *Compute capability 7.5* 는 특정 GPU(Turing 아키텍처)의 하드웨어 기능 수준을 나타낸다.

CUDA 플랫폼은 여러 세대의 GPU 아키텍처(심지어 아직 등장하지 않은 아키텍처까지)를 지원하도록 설계되어 있다. 따라서 새로운 CUDA 나오면 보통 새로운 GPU 아키텍처에 대한 지원이 추가되지만, 동시에 하드웨어 세대와 무관한 소프트웨어 기능도 포함된다.

마지막으로, 과거 아키텍처인 **Tesla** 와 **Fermi** 는 각각 **CUDA 7.0** 과 **CUDA 9.0** 부터 공식 지원이 종료되었다.
