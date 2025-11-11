## 5.1 커널의 내부 동작 (VecAdd 예제 중심)

### 커널 호출 시점: CPU → GPU 명령 전달

```cpp
VecAdd<<<1, N>>>(A, B, C);
```

이 문장은 CPU 입장에서 “GPU야, `VecAdd` 커널을 한 블록(block) 안에 N개의 스레드(thread)로 실행해줘”라는 요청이다.

* CPU의 명령은 CUDA 런타임을 통해 **드라이버 계층**으로 전달되고,
  드라이버는 이 명령을 GPU의 **명령 큐(Command Stream)**에 push한다.
* 실제 GPU는 이 시점에서 커널의 코드(PTX → SASS로 컴파일된 형태)와
  실행 구성을 함께 받아 **디바이스 메모리(Global Memory)**에 로드한다.
* 이후 GPU의 **GigaThread Engine**이 이 커널을 SM 단위로 분배(scheduling)한다.

즉, 호출 직후 CPU는 블로킹되지 않고 다음 코드를 이어서 실행할 수 있으며,
GPU 내부에서는 이 명령이 **비동기적으로 스케줄링 준비 단계**에 들어간다.

---

### Grid, Block 생성 및 스케줄링

`<<<1, N>>>`이라는 실행 구성은 다음 의미를 가진다:

* **Grid:** 1개
* **Block:** 1개
* **Thread:** N개

GPU의 **GigaThread Engine**은 Grid 내 Block 단위로 일을 나눈다.
이 예제에서는 Block이 1개뿐이므로, 이 Block이 SM 중 하나에 배치된다.

* GPU는 보통 수십~수백 개의 SM을 가지며,
  각 SM은 동시에 여러 Block을 수용할 수 있다.
* 여기서는 Block 1개이므로 단일 SM에 배치된다.

즉, **하나의 SM이 전체 VecAdd 작업을 담당**하게 된다.

---

### SM 내부: 스레드 분할 및 Warp 생성

SM은 Block 내부의 N개의 스레드를 **Warp 단위(32 threads)**로 묶는다.

예를 들어 `N = 256`이면 다음과 같은 구성이 된다.

| Warp ID | 포함된 threadIdx.x 범위 |
| ------- | ------------------ |
| Warp 0  | 0 ~ 31             |
| Warp 1  | 32 ~ 63            |
| Warp 2  | 64 ~ 95            |
| ...     | ...                |
| Warp 7  | 224 ~ 255          |

각 Warp는 SM 내부의 **Warp Scheduler**에 의해 관리된다.
이 스케줄러는 SM에 있는 여러 Warp 중 **ready 상태**인 Warp를 하나씩 선택해 명령을 발행(issue)한다.
즉, 256개의 스레드가 동시에 실행되는 게 아니라,
Warp 단위로 교차 발행(interleaving)되며 **latency hiding**을 수행한다.

---

### 스레드 실행 준비: 레지스터 할당 및 명령 디코딩

`__global__ void VecAdd(...)` 커널이 실제로 실행되기 전,
각 스레드에는 다음 리소스가 할당된다.

* **레지스터 파일(Register File):** 각 스레드별로 독립된 레지스터 공간 확보
* **공유 메모리(Shared Memory):** Block 단위로 공유 공간 확보 (여기서는 사용 X)
* **Thread Context:** threadIdx, blockIdx 등 내장 변수 초기화

예제 코드에서 `int i = threadIdx.x;`는
각 스레드의 **Thread Context**에서 자신의 x 인덱스를 읽어오는 단순한 MOV 명령이다.

즉, 이 시점에 스레드별 인덱스가 다음과 같이 설정된다.

| Thread ID | i 값 | 의미                       |
| --------- | --- | ------------------------ |
| 0         | 0   | C[0] = A[0] + B[0]       |
| 1         | 1   | C[1] = A[1] + B[1]       |
| ...       | ... | ...                      |
| N-1       | N-1 | C[N-1] = A[N-1] + B[N-1] |

---

### 실제 연산: Global Memory → Register → ALU → Global Memory

이제 본격적으로 **연산 파이프라인**이 동작한다.

```cpp
C[i] = A[i] + B[i];
```

이 한 줄은 실제로 다음 순서의 마이크로 단계로 분해된다:

1. **Load (LDG)**

   * 스레드가 전역 메모리(Global Memory)에 있는 `A[i]`와 `B[i]`를 읽는다.
   * SM의 **Load/Store Unit**(LD/ST)에서 수행된다.
   * 여러 스레드가 연속된 주소를 접근하므로, **메모리 Coalescing**이 발생한다.
     → 한 번의 DRAM burst로 여러 스레드의 데이터를 동시에 가져올 수 있다.

2. **Compute (FADD)**

   * 읽어온 두 값을 SM 내부의 **FP32 ALU 파이프라인**에서 더한다.
   * 이 연산은 Tensor Core가 아닌 일반 CUDA Core(Scalar ALU)에서 수행된다.
   * Warp 내의 32개 스레드는 동일한 FADD 명령을 한 번에 실행(SIMT 방식).

3. **Store (STG)**

   * 결과값 `C[i]`를 다시 Global Memory에 기록한다.
   * Store 명령 역시 Coalescing되어 한 번의 메모리 트랜잭션으로 처리된다.

즉, 스레드 하나당 “읽기 → 더하기 → 쓰기”의 세 단계를 거치며,
이 모든 과정은 **Warp 단위로 병렬 파이프라인 처리**된다.

---

### 메모리 트랜잭션의 실제 모습

만약 `N = 256`, `A`, `B`, `C`가 모두 연속된 float 배열이라면,
Warp 0 (threadIdx 0~31)의 로드 명령은 다음과 같이 수행된다.

| ThreadIdx.x | Load Address (A) | Memory Transaction |
| ----------- | ---------------- | ------------------ |
| 0           | A[0]             | Row burst 시작       |
| 1           | A[1]             | 同 burst에 포함        |
| …           | A[31]            | 同 burst에 포함        |

→ 즉, 32개의 스레드가 32개의 연속된 float(4B) 주소를 접근하므로
→ 총 128B(=32×4B) 크기의 **한 번의 DRAM burst**로 데이터를 불러올 수 있다.
이를 **메모리 coalescing**이라고 하며, GPU 성능 최적화의 핵심 개념이다.

---

### Warp Scheduler의 역할 (Latency Hiding)

한 Warp가 Global Memory로부터 데이터를 읽는 동안(약 400~800 cycle 지연),
SM은 이 Warp를 **suspend**하고,
메모리 접근이 끝난 다른 Warp를 **즉시 실행**시킨다.

이것이 GPU의 대표적 특징인 **Zero-Overhead Context Switching**이다.
즉, SM 내부에는 항상 수십 개의 Warp가 대기 중이며,
메모리 지연이 발생할 때마다 즉시 다른 Warp를 실행해 파이프라인을 멈추지 않는다.

→ 이 덕분에 GPU는 메모리 지연(latency)을 느끼지 않고
항상 연산 장치를 “가득 채운 상태”로 유지할 수 있다.

---

### 커널 종료와 결과 수집

모든 스레드가 자신의 연산을 완료하면,
SM은 이 Block의 상태를 **complete**로 표시하고 GigaThread Engine에 보고한다.
Grid 내 모든 Block이 완료되면,
커널 전체가 종료되고 GPU는 CPU로 제어권을 반환한다.

CPU는 이후 다음 단계에서 결과를 가져온다:

```cpp
cudaMemcpy(C_host, C_device, N*sizeof(float), cudaMemcpyDeviceToHost);
```

이때 **PCIe 또는 NVLink**를 통해 GPU 메모리(Global Memory)에서
CPU의 메인 메모리로 결과가 복사된다.

---

### 요약: 내부 동작 흐름

| 단계            | GPU 내부에서 일어나는 일      | 주요 하드웨어 단위           |
| ------------- | -------------------- | -------------------- |
| 커널 호출         | 명령 큐에 커널 등록          | GigaThread Engine    |
| Block 배치      | Block → SM 배정        | SM Dispatcher        |
| 스레드 생성        | N threads → Warps 구성 | Warp Scheduler       |
| threadIdx 초기화 | Thread Context 생성    | Register File        |
| Load          | A[i], B[i] 읽기        | LD/ST Unit, L2, DRAM |
| Compute       | A[i] + B[i]          | FP32 ALU (CUDA Core) |
| Store         | 결과 C[i] 저장           | LD/ST Unit           |
| Warp 교체       | 메모리 지연 중 다른 Warp 실행  | Warp Scheduler       |
| 커널 완료         | Block 완료 보고, Grid 종료 | GigaThread Engine    |

---

### 결론적으로

이 단순한 한 줄짜리 연산:

```cpp
C[i] = A[i] + B[i];
```

은 GPU 내부에서 다음과 같은 거대한 병렬 파이프라인을 거친다.

1. CPU의 명령이 GPU 명령 큐로 전송
2. SM 단위로 Block이 배치
3. Warp 단위로 스레드가 그룹화
4. Coalesced 메모리 접근으로 효율적인 Global Memory 사용
5. Warp Scheduler가 메모리 지연을 숨기며 교차 실행
6. FP32 CUDA Core에서 연산
7. 결과를 Global Memory에 저장

결국 한 줄의 덧셈이 **수천 개의 병렬 스레드**,
**다단계 메모리 계층**, **Warp 기반 스케줄링 구조**를 거쳐
완전히 병렬적으로 수행되는 것이다.
