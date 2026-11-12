"""FastAPI 앱: POST /query 와 테스트용 웹 화면(web/)을 제공한다.

[Java 개발자를 위한 개요]
- 이 파일은 Spring Boot 의 @RestController + application 설정 역할을 한다.
- FastAPI 는 데코레이터(@app.post)로 URL 과 함수를 연결한다. Spring 의 @PostMapping 과 같다.
- 요청/응답 본문은 pydantic BaseModel 로 정의한다. Java 의 DTO(record/Lombok @Data) + Bean Validation 과 비슷하다.
- 실행: `.venv\\Scripts\\python.exe -m uvicorn src.api:app --port 8765`
  (uvicorn 은 Tomcat 같은 웹 서버, `src.api:app` 은 "src/api.py 파일의 app 변수" 라는 뜻)
"""

# ── 표준 라이브러리 import ─────────────────────────────────────────
# Python 의 import 는 Java 의 import 와 같지만, 모듈(파일) 단위로 가져온다.
import threading  # 스레드 잠금 (java.util.concurrent.locks 와 비슷)
import uuid  # UUID 생성 (java.util.UUID)
from pathlib import Path  # 파일 경로 객체 (java.nio.file.Path)
from typing import Literal  # 정해진 문자열 값만 허용하는 타입 (Java enum 과 비슷한 용도)

from dotenv import load_dotenv

# .env 파일의 값(AWS 키, LangSmith 설정)을 환경변수로 읽어 들인다.
# 아래 import 들이 환경변수를 읽을 수 있으므로 다른 import 보다 먼저 실행한다.
# (Python 은 import 문도 위에서 아래로 "실행"되는 코드라서 순서가 의미를 가진다)
load_dotenv()

# noqa: E402 는 "import 가 파일 맨 위에 있지 않다"는 린터 경고를 끄는 표시다. 위 load_dotenv() 때문에 일부러 아래에 둔다.
from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402  AWS SDK 예외 클래스
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402  정적 파일(html, css, 이미지) 제공
from pydantic import BaseModel, Field  # noqa: E402  DTO 정의와 값 검증

from src import agent  # noqa: E402  우리 프로젝트의 src/agent.py 모듈

# __file__ 은 지금 이 파일(src/api.py)의 경로다. parent 를 두 번 타면 프로젝트 루트, 거기에 "web" 을 붙인다.
# Path 객체는 / 연산자로 경로를 이어 붙일 수 있다 (Java 의 path.resolve("web") 과 같다).
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 응답 status 로 허용하는 값 4개. 이 밖의 값을 넣으면 pydantic 이 검증 오류를 낸다.
Status = Literal["collecting", "pending_confirmation", "submitted", "cancelled"]


# ── 요청·응답 DTO ─────────────────────────────────────────────────
# class X(BaseModel): 는 "BaseModel 을 상속한 클래스 X" 라는 뜻이다 (Java 의 class X extends BaseModel).
# 클래스 본문에 `필드명: 타입` 을 쓰면 그대로 필드가 되고, 생성자·getter·JSON 변환이 자동으로 만들어진다.


class QueryRequest(BaseModel):
    """POST /query 요청 본문. 예: {"question": "준비물이 뭐예요?", "thread_id": "abc..."}"""

    # Field(min_length=1): 빈 문자열이면 422 오류 (Java 의 @NotBlank/@Size(min=1) 과 비슷)
    question: str = Field(min_length=1)
    # `str | None = None` 은 "문자열 또는 null, 기본값 null" 이라는 뜻 (Java 의 @Nullable String threadId = null)
    thread_id: str | None = None


class Context(BaseModel):
    """RAG 검색 근거 한 조각. doc_id 는 규정 문서 번호(예: REG-03), text 는 검색된 문단."""

    doc_id: str
    text: str


class TraceStep(BaseModel):
    """실행 기록 한 단계. 어떤 단계(step)에 무엇을 넣고(input) 무엇이 나왔는지(output)."""

    step: str
    input: str
    output: str


class QueryResponse(BaseModel):
    """POST /query 응답 본문 (제출 규약이라 키 이름을 바꾸면 안 된다)."""

    answer: str
    contexts: list[Context]  # list[Context] 는 Java 의 List<Context>
    trace: list[TraceStep]
    thread_id: str
    status: Status


# FastAPI 애플리케이션 객체. Spring 의 ApplicationContext 처럼 라우트(URL)를 모아 두는 곳이다.
app = FastAPI(title="사업장 방문 신청 어시스턴트")

# 서버가 발급한 thread_id 목록 (Java 의 Set<String>).
# 메모리에만 있으므로 서버를 재시작하면 사라진다. 대화 상태를 담는 MemorySaver 와 수명이 같다.
# 이름 앞의 밑줄(_)은 "이 모듈 안에서만 쓰는 값" 이라는 관례다 (Java 의 private 에 해당하지만 강제는 아님).
_threads: set[str] = set()

# 대화(thread_id)별 잠금 (Java 의 ConcurrentHashMap<String, Lock> 과 같은 용도).
# FastAPI 는 `def` 로 정의한 핸들러를 스레드 풀에서 실행하므로, 같은 대화의 요청 두 개가 동시에 들어올 수 있다
# (예: 확정 버튼 두 번 누르기). 이 잠금으로 같은 대화의 요청을 한 번에 하나씩, 도착 순서대로 처리한다.
# 두 번째 요청은 거절되지 않고 첫 요청이 끝날 때까지 기다린 뒤 처리된다 → 이미 접수된 상태를 보고 같은 신청번호를 재안내한다.
# 서로 다른 대화는 각자 다른 잠금을 쓰므로 서로 기다리지 않는다. (uvicorn 워커 1개 = 프로세스 1개 전제)
_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()  # _thread_locks dict 자체를 여러 스레드가 동시에 고치지 않게 보호하는 잠금


def _lock_for(thread_id: str) -> threading.Lock:
    """thread_id 전용 잠금을 돌려준다. 처음 보는 thread_id 면 새로 만든다."""
    with _thread_locks_guard:
        # setdefault(키, 기본값): 키가 있으면 그 값을, 없으면 기본값을 넣고 돌려준다 (Java 의 map.computeIfAbsent)
        return _thread_locks.setdefault(thread_id, threading.Lock())


def _resolve_thread(thread_id: str | None) -> str:
    """요청의 thread_id 를 확인한다.

    - thread_id 가 없으면(None) 새로 발급해 등록하고 돌려준다.
    - 서버가 발급한 적 없는 값이면 404 를 낸다(예외를 던지면 FastAPI 가 HTTP 응답으로 바꿔 준다).
    - 아는 값이면 그대로 돌려준다.
    """
    # Python 의 None 은 Java 의 null 이다. `is None` 으로 비교한다.
    if thread_id is None:
        new_id = uuid.uuid4().hex  # 하이픈 없는 32자리 UUID 문자열
        _threads.add(new_id)
        return new_id
    if thread_id not in _threads:  # `not in` 은 Java 의 !set.contains(x)
        # raise 는 Java 의 throw. HTTPException 은 FastAPI 가 상태 코드 응답으로 바꿔 준다.
        raise HTTPException(status_code=404, detail="알 수 없는 thread_id 입니다. 새 대화를 시작해 주세요.")
    return thread_id


# @app.post(...) 데코레이터: 아래 함수를 POST /query 핸들러로 등록한다 (Spring 의 @PostMapping("/query")).
# response_model=QueryResponse 를 주면 반환값을 이 형식으로 검증하고 JSON 으로 바꿔 준다.
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """질문을 에이전트에 넘겨 답을 돌려준다. Bedrock 타임아웃·호출 실패는 503 으로 응답한다.

    FastAPI 는 요청 JSON 을 QueryRequest 로 자동 변환해 req 로 넘겨준다 (Spring 의 @RequestBody).
    """
    thread_id = _resolve_thread(req.thread_id)
    try:
        # 같은 대화의 요청은 잠금 안에서 하나씩 처리한다 (with = 자동으로 잠그고 푸는 블록, Java 의 synchronized 블록)
        with _lock_for(thread_id):
            # agent.answer 는 {"answer": ..., "contexts": [...], "trace": [...], "status": ...} 형태의 dict(Java 의 Map)를 돌려준다.
            result = agent.answer(req.question, thread_id)
    # 여러 예외를 한 번에 잡을 때는 튜플로 쓴다 (Java 의 catch (BotoCoreError | ClientError exc)).
    # BotoCoreError: 타임아웃·연결 실패, ClientError: Bedrock 이 돌려준 오류(권한, 사용량 한도 등)
    except (BotoCoreError, ClientError) as exc:
        # `raise ... from exc` 는 원래 예외를 원인(cause)으로 연결한다 (Java 의 new Exception(msg, cause)).
        raise HTTPException(status_code=503, detail="잠시 후 다시 시도해 주세요.") from exc
    # **result 는 dict 를 풀어서 키워드 인자로 넘기는 문법이다.
    # 즉 QueryResponse(thread_id=..., answer=result["answer"], contexts=result["contexts"], ...) 와 같다.
    return QueryResponse(thread_id=thread_id, **result)


# web/ 폴더를 "/" 경로에 정적 파일로 연결한다. html=True 이면 "/" 요청에 index.html 을 돌려준다.
# 라우트는 등록 순서대로 매칭되므로, /query 를 먼저 등록하고 정적 파일은 맨 마지막에 붙인다.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
