"""POST /query 계약 테스트 (AC2). 에이전트는 가짜 모델로 바꿔 Bedrock 을 부르지 않는다.

TestClient 는 실제 서버를 띄우지 않고 FastAPI 앱에 HTTP 요청을 보내는 도구다 (Spring 의 MockMvc 와 같은 역할).
"""

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src import agent
from src.api import app
from tests.conftest import ScriptedChatModel, extract_model, guard_model, tool_call

client = TestClient(app)


# autouse=True: 이 파일의 모든 테스트 앞에 자동으로 실행된다 (인자에 적지 않아도 된다).
# 기본값으로 "도구 없이 바로 답하는 가짜 모델" 을 끼워, 어떤 테스트도 실수로 Bedrock 을 부르지 않게 한다.
@pytest.fixture(autouse=True)
def plain_answer(use_model):
    """기본은 도구 없이 바로 답하는 가짜 모델."""
    return use_model([AIMessage("가짜 답변입니다.")])


def _throttled() -> ClientError:
    """Bedrock 사용량 한도 초과(ThrottlingException) 오류 객체를 만든다."""
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "Too many tokens per day"}}, "Converse")


def test_응답에_다섯_키가_있다():
    """answer, contexts, trace, thread_id, status 를 모두 돌려준다(제출 규약)."""
    res = client.post("/query", json={"question": "안녕하세요"})
    assert res.status_code == 200
    body = res.json()  # 응답 JSON 을 dict 로 변환
    assert set(body) == {"answer", "contexts", "trace", "thread_id", "status"}
    assert body["status"] in {"collecting", "pending_confirmation", "submitted", "cancelled"}


def test_thread_id_가_없으면_새로_발급하고_재사용할_수_있다():
    """첫 요청에서 받은 thread_id 로 다시 보내면 같은 스레드로 처리한다."""
    first = client.post("/query", json={"question": "안녕하세요"}).json()
    second = client.post("/query", json={"question": "다시", "thread_id": first["thread_id"]})
    assert second.status_code == 200
    assert second.json()["thread_id"] == first["thread_id"]


def test_서버가_모르는_thread_id_는_404():
    """발급하지 않은 thread_id 는 거절한다."""
    res = client.post("/query", json={"question": "안녕하세요", "thread_id": "unknown"})
    assert res.status_code == 404


def test_규정_질문은_contexts_와_trace_를_채운다(use_model):
    """search_regulations 를 호출하면 검색 근거가 contexts 에, 도구 호출이 trace 에 담긴다."""
    # 가짜 모델 시나리오: 1번째 호출은 "search_regulations 를 불러줘", 2번째 호출은 최종 답변
    use_model([tool_call("search_regulations", {"query": "노트북 반입"}), AIMessage("반입 확인을 받으면 됩니다. (근거: REG-03)")])
    body = client.post("/query", json={"question": "노트북 가져가도 되나요?"}).json()
    assert body["answer"].endswith("(근거: REG-03)")
    assert "REG-03" in [c["doc_id"] for c in body["contexts"]]
    assert [t["step"] for t in body["trace"]] == ["입력 가드레일", "신청 정보 추출", "search_regulations", "답변 생성"]


def test_Bedrock_타임아웃은_503(use_model):
    """모델 호출이 타임아웃되면 503 으로 응답한다(다른 모델로 넘기지 않는다)."""
    use_model([ReadTimeoutError(endpoint_url="https://bedrock")])
    res = client.post("/query", json={"question": "준비물이 뭐예요?"})
    assert res.status_code == 503


def test_사용량_한도에_걸리면_다음_모델로_답한다(fake_index):
    """기본 모델이 ThrottlingException 이면 대체 모델 답을 돌려준다."""
    agent.set_agent(
        agent.build_agent(ScriptedChatModel(responses=[_throttled()]), fallbacks=[ScriptedChatModel(responses=[AIMessage("대체 모델 답변")])],
            guard_models=[guard_model()],
            extract_models=[extract_model()],
        )
    )
    body = client.post("/query", json={"question": "안녕하세요"}).json()
    assert body["answer"] == "대체 모델 답변"


def test_모든_모델이_한도에_걸리면_503(fake_index):
    """대체 모델까지 모두 한도 초과면 503 으로 응답한다."""
    agent.set_agent(
        agent.build_agent(ScriptedChatModel(responses=[_throttled()]), fallbacks=[ScriptedChatModel(responses=[_throttled()])], guard_models=[guard_model()], extract_models=[extract_model()])
    )
    res = client.post("/query", json={"question": "안녕하세요"})
    assert res.status_code == 503


def test_화면을_루트에서_제공한다():
    """GET / 은 web/index.html 을 돌려준다."""
    res = client.get("/")
    assert res.status_code == 200
    assert "방문 신청 어시스턴트" in res.text


def _basic(password: str) -> dict:
    """Basic 인증 헤더를 만든다. 사용자 이름은 아무 값이나 된다."""
    import base64  # 이 헬퍼에서만 쓰는 import (Java 와 달리 함수 안에서도 import 할 수 있다)

    token = base64.b64encode(f"guest:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_접속_비밀번호가_설정되면_없거나_틀린_요청은_401(monkeypatch):
    """ACCESS_PASSWORD 가 있으면 화면과 /query 모두 비밀번호 없이는 401 이고, 브라우저 로그인 창을 띄우는 헤더를 준다."""
    # monkeypatch.setenv: 이 테스트 동안만 환경변수를 바꾸고 끝나면 되돌린다
    monkeypatch.setenv("ACCESS_PASSWORD", "demo-pass")
    for res in (
        client.post("/query", json={"question": "안녕하세요"}),
        client.post("/query", json={"question": "안녕하세요"}, headers=_basic("wrong")),
        client.get("/"),
    ):
        assert res.status_code == 401
        assert res.headers["WWW-Authenticate"].startswith("Basic")


def test_접속_비밀번호가_맞으면_그대로_응답한다(monkeypatch):
    """비밀번호가 맞으면 /query 형식은 그대로다."""
    monkeypatch.setenv("ACCESS_PASSWORD", "demo-pass")
    res = client.post("/query", json={"question": "안녕하세요"}, headers=_basic("demo-pass"))
    assert res.status_code == 200
    assert set(res.json()) == {"answer", "contexts", "trace", "thread_id", "status"}
    assert client.get("/", headers=_basic("demo-pass")).status_code == 200


def test_접속_비밀번호가_없으면_검사하지_않는다(monkeypatch):
    """로컬 개발처럼 ACCESS_PASSWORD 가 비어 있으면 인증 없이 동작한다."""
    monkeypatch.delenv("ACCESS_PASSWORD", raising=False)
    assert client.post("/query", json={"question": "안녕하세요"}).status_code == 200
