"""테스트 공용 가짜 모델과 픽스처. Bedrock 을 호출하지 않고 결과를 고정한다(Seed 제약: 단위 테스트는 가짜 LLM·임베딩).

[Java 개발자를 위한 pytest 개요]
- conftest.py 는 pytest 가 자동으로 읽는 공용 설정 파일이다. 여기 정의한 픽스처는 import 없이 모든 테스트에서 쓸 수 있다.
- 픽스처(@pytest.fixture)는 JUnit 의 @BeforeEach/@AfterEach 를 합친 것이다.
  테스트 함수의 인자 이름이 픽스처 이름과 같으면 pytest 가 알아서 실행해 값을 넣어 준다(이름 기반 의존성 주입).
  픽스처 안의 `yield` 앞은 준비(setup), 뒤는 정리(teardown) 코드다.
- 테스트 함수 이름은 test_ 로 시작해야 하고, 검증은 assert 문 하나로 한다(JUnit 의 assertEquals 등 대신).
- 실행: `.venv\\Scripts\\python.exe -m pytest -q tests/`
"""

import hashlib  # 해시 함수 (java.security.MessageDigest)
import math

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src import agent, retriever, tools


class KeywordEmbeddings(Embeddings):
    """글자 2-gram(연속한 두 글자)을 해시해 만든 결정적 가짜 임베딩. 겹치는 단어가 많을수록 벡터가 가깝다.

    실제 Titan 임베딩처럼 의미를 이해하지는 못하지만, "노트북 반입" 처럼 같은 단어가 들어간 문서를 찾는 데는 충분하고
    네트워크 없이 항상 같은 결과를 낸다. (Java 로 치면 Embeddings 인터페이스의 테스트용 Stub 구현)
    """

    dim = 512  # 벡터 차원. 클래스 본문에 바로 쓴 변수는 클래스 변수다 (Java 의 static 필드와 비슷)

    def _embed(self, text: str) -> list[float]:
        """한 문장을 512차원 벡터로 바꾼다. 각 2-gram 을 해시해 해당 칸의 값을 1 올리고, 마지막에 길이를 1로 맞춘다."""
        vec = [0.0] * self.dim  # 0.0 이 512개인 리스트 (Java 의 new double[512])
        compact = text.replace(" ", "")
        # range(n) 은 0..n-1 정수열 (Java 의 for (int i = 0; i < n; i++))
        for i in range(len(compact) - 1):
            # compact[i : i + 2] 는 i 부터 두 글자를 잘라낸 문자열 (Java 의 substring(i, i + 2))
            h = int(hashlib.md5(compact[i : i + 2].encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        # 벡터 길이(노름)로 나눠 정규화한다. 길이가 0이면 0으로 나누지 않게 1.0 을 쓴다.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 여러 개를 벡터로 바꾼다 (Embeddings 인터페이스 구현)."""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """질의 하나를 벡터로 바꾼다 (Embeddings 인터페이스 구현)."""
        return self._embed(text)


class ScriptedChatModel(BaseChatModel):
    """정해 둔 응답을 순서대로 돌려주는 가짜 채팅 모델 (Mockito 의 when(...).thenReturn(a, b, c) 와 같은 역할).

    - responses 에 AIMessage 를 넣으면 호출될 때마다 하나씩 돌려준다. 다 쓰면 마지막 응답을 반복한다.
    - responses 에 예외 객체를 넣으면 그 차례에 예외를 던진다(타임아웃, 사용량 한도 상황 재현용).
    - BaseChatModel 은 pydantic 모델이라, 필드를 클래스 본문에 `이름: 타입` 으로 선언하면 생성자 인자가 된다.
      사용 예: ScriptedChatModel(responses=[AIMessage("안녕")])
    """

    responses: list
    calls: int = 0  # 지금까지 호출된 횟수 (테스트에서 "LLM 을 몇 번 불렀는지" 확인할 때도 쓴다)
    seen: list = []  # 호출마다 받은 메시지 목록 (모델에 무엇이 전달됐는지 확인할 때 쓴다. pydantic 이 인스턴스마다 새 리스트로 복사한다)

    # @property: 메서드를 필드처럼 읽게 만든다 (Java 의 getter 를 model.llmType 처럼 부르는 것과 같다)
    @property
    def _llm_type(self) -> str:
        """모델 종류 이름 (BaseChatModel 이 요구하는 추상 멤버 구현)."""
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        """도구 목록을 모델에 연결하는 메서드. 가짜 모델은 도구 정보가 필요 없으므로 무시하고 자기 자신을 돌려준다.

        **kwargs 는 "이름이 정해지지 않은 키워드 인자를 dict 로 모두 받는다" 는 뜻이다 (Java 의 Map<String, Object> options 와 비슷).
        """
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """모델 호출 1회. 다음 응답을 꺼내 돌려준다 (BaseChatModel 이 요구하는 추상 메서드 구현)."""
        # min(...) 으로 인덱스가 목록 끝을 넘지 않게 한다 → 다 쓰면 마지막 응답 반복
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1  # Python 에는 ++ 연산자가 없다
        self.seen.append(messages)
        if isinstance(response, Exception):
            raise response
        # 같은 응답 객체를 다시 쓰면 메시지 id 가 같아서, 대화 기록에 새로 붙지 않고 앞 메시지를 덮어쓴다(실제 모델은 매번 새 id).
        # 그래서 id 를 비운 복사본을 돌려준다 (Java 의 clone() 후 id = null). 비운 id 는 LangChain 이 새로 채운다.
        return ChatResult(generations=[ChatGeneration(message=response.model_copy(update={"id": None}))])


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    """"이 도구를 이 인자로 불러 달라" 는 LLM 응답(도구 호출 메시지)을 만든다. ScriptedChatModel 의 응답 목록에 넣어 쓴다.

    예: tool_call("search_regulations", {"query": "노트북 반입"})
    """
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def guard_model(category: str = "normal") -> ScriptedChatModel:
    """입력 가드레일 분류기 자리에 넣을 가짜 모델. 항상 주어진 category 로 분류한다.

    가드레일은 LCEL 구조화 출력(with_structured_output)을 쓰는데, 내부적으로 "GuardrailResult 도구 호출" 형태의 응답을 읽는다.
    그래서 가짜 모델도 그 도구 호출 메시지를 돌려주면 된다.
    """
    return ScriptedChatModel(responses=[tool_call("GuardrailResult", {"category": category})])


def extract_model(form: dict | None = None) -> ScriptedChatModel:
    """신청 정보 추출기 자리에 넣을 가짜 모델. 항상 주어진 form(필드 dict)을 추출 결과로 돌려준다. None 이면 아무것도 추출하지 않는다."""
    return ScriptedChatModel(responses=[tool_call("VisitForm", form or {})])


# autouse=True: 모든 테스트에 자동 적용된다. tmp_path, monkeypatch 는 pytest 가 기본 제공하는 픽스처다.
#   - tmp_path: 테스트마다 새로 만들어지는 임시 폴더 (JUnit 의 @TempDir)
#   - monkeypatch: 테스트 동안만 변수·환경변수를 바꾸고 끝나면 원래대로 되돌려 준다
@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """신청 저장소를 임시 경로로 바꾸고(작업 폴더의 data/runtime 을 건드리지 않는다), 날짜 기준일을 2026-11-12(목)로 고정한다."""
    monkeypatch.setattr(tools, "REQUESTS_PATH", tmp_path / "requests.json")
    monkeypatch.setenv("VISIT_REFERENCE_DATE", "2026-11-12")
    return tmp_path / "requests.json"


@pytest.fixture
def fake_index():
    """가짜 임베딩으로 만든 규정 인덱스를 끼워 두고, 테스트가 끝나면 비운다.

    테스트 함수에 fake_index 인자를 적으면 이 픽스처가 먼저 실행된다.
    """
    retriever.set_index(retriever.build_index(KeywordEmbeddings()))  # 준비 (@BeforeEach)
    yield  # 여기서 테스트 본문이 실행된다
    retriever.set_index(None)  # 정리 (@AfterEach): 다른 테스트에 영향이 없도록 초기화


@pytest.fixture
def use_model(fake_index):
    """가짜 모델로 에이전트를 만들어 끼워 넣는 함수를 돌려준다. 인자로 fake_index 를 받으므로 가짜 인덱스도 함께 준비된다.

    사용 예:
        def test_x(use_model):
            use_model([tool_call("search_regulations", {"query": "주차"}), AIMessage("최종 답변")])
            use_model([AIMessage("답")], guard="injection")   # 가드레일이 인젝션으로 분류하는 상황
    """

    # 함수 안에 함수를 정의할 수 있다(중첩 함수). 바깥 픽스처는 이 함수 자체를 테스트에 넘겨준다.
    # guard: 가드레일 분류기가 돌려줄 분류 (기본 normal = 통과)
    # form: 신청 정보 추출기가 돌려줄 필드 dict (기본 None = 추출 없음)
    def _use(responses: list, guard: str = "normal", form: dict | None = None) -> ScriptedChatModel:
        model = ScriptedChatModel(responses=responses)
        agent.set_agent(agent.build_agent(model, guard_models=[guard_model(guard)], extract_models=[extract_model(form)]))
        return model

    yield _use
    agent.set_agent(None)  # 정리: 다음 테스트가 이전 가짜 에이전트를 쓰지 않게 초기화
