# 사업장 방문 신청 어시스턴트

외부 방문객이 채팅으로 방문 규정을 묻고, 방문 신청과 본인 신청 조회·취소까지 한 대화에서 처리하는 Agentic RAG 에이전트입니다.
규정 답변에는 근거 문서 번호를 붙이고, 신청은 마스킹된 요약을 본인이 '확정'해야만 저장합니다.

- 왜 만드는지, 누가 쓰는지, 확장 관점과 서비스 정책: [SERVICE.md](SERVICE.md)
- 데이터(규정 문서, 임직원, 시드 신청)는 모두 익명화된 가상 데이터입니다.

## 할 수 있는 일

| 기능 | 예시 입력 | 동작 |
|---|---|---|
| 규정 안내 (RAG) | 개인 노트북을 가지고 들어가도 되나요? | FAISS 로 규정 문서를 검색해 답하고 `(근거: REG-03)` 를 붙인다. 문서에 없는 내용은 확인할 수 없다고 답한다 |
| 방문 신청 | 11월 17일 오전 10시 박지훈 님 미팅. 홍길동 010-1234-5678 한빛정밀 | 항목 수집 → 검증(과거·당일·주말, 휴대폰 형식, 피방문자) → 마스킹된 요약 → '확정' 시 저장·신청번호 발급 |
| 나눠 말하기 | "한빛정밀" → "회의" → "정민호" → "11월 17일" → "오후 2시" | 직전 질문을 보고 항목을 채우고, 날짜·시각을 나눠 받아도 합친다 |
| 확정 전 수정 | 차량번호 12가3456 추가해 주세요 | 저장 없이 바뀐 요약을 다시 보여 준다 (항목 변경, 차량번호 추가·삭제, 질문) |
| 본인 신청 조회 | 제 신청 내역 보여 주세요. 김방문 010-2222-3333 | 이름+휴대폰번호가 맞을 때만 방문일 순 목록. 신청번호는 일부만 표시 |
| 상세·취소 | 1번 신청 자세히 / 신청번호 전체 입력 | 목록 번호로 상세, 전체 신청번호가 맞으면 방문 전날까지 즉시 취소 |

## 구조

```
사용자 ─ POST /query ─▶ FastAPI (thread_id 별 잠금)
                          │
                          ▼
          create_agent (LangGraph, MemorySaver)
          ├─ 미들웨어 1  입력 가드레일   고유식별정보(정규식)·인젝션·타인 조회(LCEL 분류) 차단
          ├─ 미들웨어 2  신청 정보 추출  LCEL 구조화 출력 → 그래프 state (수집 필드의 기준값)
          ├─ 미들웨어 3  사용량 한도 대체 Sonnet 4.6 → Haiku 4.5 (ThrottlingException 일 때만)
          └─ 도구 6개
             search_regulations  규정 검색 (FAISS + Titan 임베딩)
             find_host           피방문자 확인
             create_visit_request 검증 → interrupt() 확정 대기 → '확정' 시 저장
             get_my_requests / get_request_detail / cancel_my_request  본인 조회·상세·취소
```

- 응답, trace, LangSmith 기록의 이름·휴대폰번호·차량번호는 마스킹합니다. 원본은 신청 저장소 JSON(`data/runtime/`, gitignore)에만 둡니다.
- 같은 대화의 '확정'이 두 번 와도(순차·동시) 저장은 1건이고, 같은 신청번호를 다시 안내합니다.

| 파일 | 역할 |
|---|---|
| [src/api.py](src/api.py) | FastAPI 앱. `POST /query`, `web/` 정적 화면 제공 |
| [src/agent.py](src/agent.py) | 에이전트 그래프, 미들웨어(가드레일·추출·모델 대체), 마스킹, HITL 재개 |
| [src/tools.py](src/tools.py) | 도메인 도구 6개와 검증·저장 로직 |
| [src/retriever.py](src/retriever.py) | 규정 문서 청킹·FAISS 인덱스·검색 |
| [web/](web/) | 에이전트 테스트용 채팅 화면 (시나리오 메뉴, 신청서 카드, trace 표시) |
| [data/](data/) | 가상 규정 문서 6개, 더미 임직원, 시드 신청 1건 |
| [evaluation/](evaluation/) | 평가 세트 20건, 평가 러너, round1·round2 리포트 |
| [tests/](tests/) | 단위 테스트 (가짜 LLM·임베딩 사용, Bedrock 호출 없음) |

## 실행

필요: Python 3.12, Amazon Bedrock 접근 권한(us-east-1), 선택으로 LangSmith 키

```bash
# 1. 가상환경과 패키지
uv venv --python 3.12 .venv
uv pip install -r requirements.txt        # 또는 .venv 활성화 후 pip install -r requirements.txt

# 2. 환경변수: .env.example 을 .env 로 복사하고 AWS 키를 채운다
cp .env.example .env

# 3. 서버 실행 (워커 1개 전제)
.venv/Scripts/python.exe -m uvicorn src.api:app --port 8765    # macOS/Linux: .venv/bin/python
```

브라우저에서 http://127.0.0.1:8765/ 을 열면 채팅 화면이 나옵니다. 상단 시나리오 메뉴로 평가 세트 20건을 바로 보낼 수 있고,
응답마다 근거 문서와 실행 기록(trace)을 펼쳐 볼 수 있습니다.

- 날짜 검증 기준일은 KST 오늘입니다. 재현이 필요하면 `.env` 의 `VISIT_REFERENCE_DATE=2026-11-12` 로 고정합니다(API 로는 바꿀 수 없음).
- Bedrock 호출은 타임아웃 30초·재시도 없음이고, 실패하면 503 을 돌려주며 저장하지 않습니다.

## 외부 배포 (Render)

[render.yaml](render.yaml) 로 Render 웹 서비스 1개를 만듭니다.

1. 이 저장소를 GitHub 에 올린 뒤 Render 대시보드에서 **New > Blueprint** 를 누르고 저장소를 고릅니다.
2. 처음 만들 때 비밀 값을 입력합니다: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ACCESS_PASSWORD`, (선택) `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.
   AWS 키는 Bedrock 호출 권한만 준 IAM 사용자 키를 따로 만들어 쓰고, AWS 예산 알림을 걸어 둡니다.
3. 배포가 끝나면 `https://<서비스 이름>.onrender.com` 에 접속합니다. 브라우저 로그인 창에 사용자 이름은 아무 값, 비밀번호는 `ACCESS_PASSWORD` 를 넣습니다.

- `ACCESS_PASSWORD` 가 있으면 화면과 `/query` 모두 HTTP Basic 인증을 요구합니다(없으면 401). `/query` 요청·응답 형식은 같습니다.
- 무료 플랜은 15분 동안 요청이 없으면 잠들고, 다시 깨어날 때 1분 정도 걸립니다. 시작할 때 규정 문서 인덱스를 다시 만듭니다.
- 디스크가 영구 저장소가 아니라서, 재배포나 재시작 때 새로 접수된 신청(`data/runtime/`)과 대화가 사라집니다. 시드 신청은 남습니다.
- 공개 화면에는 실제 이름·연락처를 입력하지 않도록 안내합니다.

## API

```http
POST /query
{"question": "방문할 때 준비물이 뭐예요?", "thread_id": "선택"}
```

```json
{
  "answer": "방문 당일에는 신분증과 신청번호를 준비하시면 됩니다. ... (근거: REG-02)",
  "contexts": [{"doc_id": "REG-02", "text": "..."}],
  "trace": [{"step": "입력 가드레일", "input": "사용자 입력", "output": "통과"}, "..."],
  "thread_id": "…",
  "status": "collecting"
}
```

- `status`: `collecting` / `pending_confirmation` / `submitted` / `cancelled`
- `thread_id` 를 빼면 새 대화, 서버가 모르는 `thread_id` 는 404

## 테스트와 평가

```bash
# 단위 테스트 (Bedrock 호출 없음, 신청 저장소는 임시 경로)
.venv/Scripts/python.exe -m pytest -q

# 평가 (Bedrock 호출, 기준일 2026-11-12 고정)
.venv/Scripts/python.exe -m evaluation.run_eval --round 2 --changes "변경 내역"
```

평가는 [evaluation/test_queries.csv](evaluation/test_queries.csv) 20건(positive 8 / negative 4 / edge 4 / guardrail 4)을 케이스마다 새 대화로 실행하고,
코드 규칙 검사(예상 도구, status, 원본 개인정보 노출, 확정 없는 저장)와 LLM-as-Judge(Haiku 4.5)를 모두 통과해야 PASS 로 셉니다.
RAGAS 는 규정 안내 케이스(1~5)에 대해 3회 측정 평균입니다.

| 항목 | round1 | round2 | 기준 |
|---|---|---|---|
| 통과 | 17 / 20 | 20 / 20 | 16 이상, round1 보다 많을 것 |
| 가드레일 위반 | 0 | 0 | 0 |
| RAGAS faithfulness | 0.981 | 0.92 | 0.8 이상 |
| RAGAS answer_relevancy | 0.67 | 0.70 | 0.7 이상 |

round1 → round2 변경 내역과 케이스별 결과는 [evaluation/round2_report.md](evaluation/round2_report.md) 에 있습니다.
answer_relevancy 는 기준과 같은 값이라 여유가 없고, RAGAS 점수는 같은 답변도 측정마다 흔들립니다(3회 평균을 쓰는 이유).

## 범위와 한계

- 레거시 방문 신청 시스템과 연동하지 않습니다. 도구는 더미 JSON 을 읽고 씁니다.
- 접수 후 신청 수정, 차량 출입 별도 등록, 반입 물품 신고, 단체 방문, SMS/메일 발송, 피방문자 승인은 범위 밖입니다.
- 대화 상태는 메모리(MemorySaver)에만 있어 서버를 재시작하면 대화가 사라집니다. 저장된 신청은 파일에 남습니다.
- 중복 저장 방지는 uvicorn 워커 1개(프로세스 1개)를 전제로 합니다.
- Multi-Agent Supervisor, MCP 연동, 장기 메모리 같은 확장 방향은 [SERVICE.md](SERVICE.md) 에 정리했습니다.
