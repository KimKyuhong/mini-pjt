# 프로젝트 규칙

사업장 방문 신청 어시스턴트 (Agentic RAG 미니 프로젝트).
무엇을 왜 만드는지는 SERVICE.md, 확정된 상세 스펙은 Seed `~/.ouroboros/seeds/seed_abf86f0a55b5.yaml` 을 기준으로 삼는다.

## 코딩 원칙 (반드시 준수)
아래 파일의 가이드라인을 모든 작업에서 지킨다. 이 파일의 다른 규칙과 함께 적용한다.
1. 코딩 전에 먼저 생각한다: 가정을 밝히고, 모호하면 묻는다
2. 단순하게 만든다: 요청한 것만, 필요한 만큼만 구현한다
3. 필요한 곳만 고친다: 요청과 관계없는 코드는 손대지 않는다
4. 목표를 정하고 검증한다: 성공 기준을 세우고 테스트로 확인한다

@andrej-karpathy-guideline.md

## 기술 스택
- Python, LangChain, LangGraph, FastAPI
- 모델은 Amazon Bedrock (ChatBedrockConverse, 리전 us-east-1), 모델 ID 는 코드 상수로 고정한다
  - 에이전트: us.anthropic.claude-sonnet-4-6 → global.anthropic.claude-sonnet-4-6 → us.anthropic.claude-haiku-4-5-20251001-v1:0 → global.anthropic.claude-haiku-4-5-20251001-v1:0 순서. 앞 모델이 사용량 한도(ThrottlingException)일 때만 다음 모델로 넘어간다(미들웨어). 타임아웃 등 다른 오류는 503
  - 평가(LLM-as-Judge·RAGAS): global.anthropic.claude-haiku-4-5-20251001-v1:0
  - 임베딩: amazon.titan-embed-text-v2:0
  - Claude 5 계열은 현재 키에 권한이 없어 쓰지 않는다
- Agent 생성은 langchain.agents 의 create_agent 를 쓴다. 단일 에이전트 + 미들웨어 구조
- 벡터 저장소는 로컬 FAISS, 대화 상태는 LangGraph MemorySaver
- Observability 는 LangSmith, 평가는 RAGAS + LLM-as-Judge (Bedrock 사용)
- Multi-Agent Supervisor, MCP, 장기 메모리는 쓰지 않는다 (SERVICE.md 확장 관점에만 기술)

## 폴더 구조 (제출 규약. 바꾸지 않는다)
src/agent.py       메인 에이전트 그래프, 미들웨어(가드레일·마스킹), HITL
src/tools.py       도메인 도구 6개 (search_regulations, find_host, create_visit_request, get_my_requests, get_request_detail, cancel_my_request)
src/retriever.py   RAG 파이프라인 (FAISS)
src/api.py         FastAPI 앱 (POST /query, web/ 정적 제공)
web/               에이전트 테스트용 웹 화면 (index.html, app.css, tokens.css, base.css, 사진)
data/              가상 규정 문서, 더미 임직원 JSON, 시드 신청 JSON
evaluation/        test_queries.csv 와 평가 리포트 (roundN_report.md, roundN_summary.json)
tests/             단위 테스트 (test_retriever, test_api, test_visit_flow, test_tools, test_guardrails)

## 주고받는 형식 (제출 규약)
- POST /query 로 받고 question 필드를 읽는다. thread_id 는 선택 필드
- 답은 answer, contexts, trace 세 키로 돌려준다. thread_id, status 를 함께 돌려준다
- status 는 collecting, pending_confirmation, submitted, cancelled 중 하나
- thread_id 가 없으면 새 스레드, 서버가 모르는 thread_id 는 404
- Bedrock 호출은 타임아웃 30초, 재시도 없음. 실패하면 503 을 돌려주고 저장하지 않는다

## 도메인 규칙
- 수집 필드는 이름, 휴대폰번호, 소속회사, 방문일시, 피방문자, 방문목적, (선택) 차량번호만 받는다
- 수집 필드의 기준값은 LCEL 구조화 추출 결과(그래프 state)이고, 도구는 그 값을 받아 쓴다
- 검증: 과거·당일 신청 불가(최소 1일 전), 주말 불가, 휴대폰번호 형식, 없는 피방문자 거부
- 날짜 기준일은 KST 오늘. 환경변수 VISIT_REFERENCE_DATE(YYYY-MM-DD)로만 바꿀 수 있고 API 로는 받지 않는다
- 저장 직전 interrupt() 로 마스킹된 요약을 보여주고 '확정' 일 때만 저장한다. 그 밖의 입력(항목 수정, 차량번호 추가·삭제, 질문)은 저장 없이 확인 단계를 닫고 일반 턴(가드레일·추출)으로 처리한 뒤 바뀐 요약을 다시 보여 준다. 수정값이 검증에 실패하면 오류를 안내하고 수집 상태로 돌아간다. 범위 밖인 '신청 수정'은 접수(submitted) 뒤의 수정을 뜻한다
- 접수(확정) 뒤에는 대화 state 의 신청 항목을 비운다. 같은 대화의 새 신청은 처음부터 다시 받아 검증·확인 단계를 거친다
- 이미 submitted 인 신청에 '확정' 이 다시 오면 같은 신청번호를 다시 안내하고 저장하지 않는다. 이 판별은 LLM 호출 전 state 검사로 한다
- 중복 저장 방지: 같은 thread_id 의 /query 는 thread_id 별 잠금으로 직렬화하고(uvicorn 단일 워커), 확정 턴은 저장을 마지막 단계로 둬 저장 뒤 LLM 을 부르지 않는다(Command goto=END). 저장소 쓰기는 프로세스 잠금 안에서 일련번호로 발급한다. 확정과 취소의 동시 도착은 범위 밖
- 본인 신청 목록·상세는 이름+휴대폰번호가 함께 맞을 때만 보여 준다(지난 방문·취소 건 포함, 방문일 순, 신청번호는 일부만 표시). 불일치·0건은 존재 여부를 드러내지 않는 같은 문구로 답한다
- 한 대화에서는 처음 확인된 본인만 조회한다. 다른 이름·휴대폰번호로 조회하려 하면 새 대화로 안내한다
- 신청 취소는 방문일이 기준일(KST 오늘)보다 뒤인 접수 건만, 전체 신청번호까지 맞을 때 확인 단계 없이 즉시 처리한다. 삭제하지 않고 status=cancelled·cancelled_at 으로 남기며, 시드 신청은 파일을 바꾸지 않고 런타임 저장소 기록이 우선한다
- 조회·거절 응답은 대화 status 를 바꾸지 않고, 취소 성공 응답만 cancelled 로 돌려준다

## 보안·개인정보
- 원본 개인정보는 신청 저장소 JSON 에만 둔다. 응답, trace, 로그, LangSmith, 평가 리포트에서는 마스킹한다
- 주민등록번호·여권번호·카드번호는 받지 않는다
- 실행 중 생기는 신청 저장소 파일은 gitignore 한다. data/ 에는 실제 사내 정보·연락처·내부 명칭을 넣지 않는다

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다
- 함수와 도구에는 한국어 docstring 을 쓴다
- 사용자는 Java 개발자이고 파이썬 문법에 익숙하지 않다. 모든 파이썬 코드(src, tests, evaluation)에 친절한 주석을 단다
  - 파일 맨 위 docstring 에 이 파일의 역할과 흐름을 Java 개념에 빗대어 설명한다 (예: FastAPI 라우트 = @PostMapping, 픽스처 = @BeforeEach)
  - 파이썬 특유 문법(컴프리헨션, 슬라이싱, 언패킹, **kwargs, with, yield, 데코레이터, global 등)이 처음 나오는 곳에 Java 대응 표현을 주석으로 단다
  - 함수 docstring 에 Args/Returns 와 "왜 이렇게 했는지"를 적는다. 동작 변경 없이 주석만 고칠 때도 테스트를 다시 돌려 확인한다
- 비밀 값은 .env 에서 읽고 코드에 적지 않는다. 새 환경변수는 .env.example 에 추가한다
- 테스트는 신청 저장소를 임시 경로로 바꿔 작업 폴더 파일을 건드리지 않는다

## 응답 언어
- 모든 응답은 한국어(한글)로 작성한다. 설명, 질문, 요약은 항상 한국어로 쓴다
- 코드, 명령어, 파일 경로, 식별자, 오류 메시지 원문은 그대로 둔다

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다
- 범위에서 제외한 기능(접수 후 신청 수정, 차량 출입 등록, 반입 물품 신고, 단체 방문, SMS/메일 발송, 피방문자 승인)을 만들지 않는다
- 폴더 구조와 /query 형식을 임의로 바꾸지 않는다
