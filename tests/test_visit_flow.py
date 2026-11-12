"""방문 신청 흐름 테스트 (AC3).

- 3단계: 수집·검증 → 확정 대기(HITL) → 확정 저장 / 취소 / 재질문
- 4단계: 중복 저장 방지 (순차 재확정, 겹쳐 온 확정, 서로 다른 대화의 동시 저장)
저장소는 conftest.py 의 isolated_store 픽스처가 임시 파일로 바꿔 두고, 날짜 기준일은 2026-11-12(목)로 고정된다.
"""

import json
import re
import threading  # 동시 요청을 만들기 위한 스레드 (java.lang.Thread)
import time

from botocore.exceptions import ReadTimeoutError
from fastapi.testclient import TestClient

from src import tools
from src.api import app
from tests.conftest import tool_call

client = TestClient(app)

# 시나리오 6번과 같은 정상 신청 정보 (2026-11-17 은 화요일, 기준일 다음 주)
FORM = {
    "visitor_name": "홍길동",
    "phone": "010-1234-5678",
    "company": "한빛정밀",
    "visit_datetime": "2026-11-17T10:00",
    "host": "박지훈",
    "purpose": "설비 미팅",
}
# 에이전트 모델 시나리오: 신청 정보가 모이면 create_visit_request 를 인자 없이 호출한다
APPLY = [tool_call("create_visit_request", {}, "call-1")]
RAW_PHONE = re.compile(r"01[016789]-?\d{3,4}-?\d{4}")


def _post(question: str, thread_id: str | None = None) -> dict:
    """질문을 보내고 응답 JSON 을 돌려준다. thread_id 를 주면 같은 대화로 이어 보낸다."""
    payload = {"question": question}
    if thread_id:
        payload["thread_id"] = thread_id
    res = client.post("/query", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


def _stored(path) -> list:
    """임시 신청 저장소의 내용을 읽는다. 파일이 없으면 빈 목록."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ── 확정 대기 → 확정 / 취소 / 재질문 ───────────────────────────────


def test_신청하면_마스킹된_요약과_함께_확정_대기로_멈추고_저장하지_않는다(use_model, isolated_store):
    """필수 정보가 모이면 status=pending_confirmation, 요약은 마스킹, 저장은 0건."""
    use_model(APPLY, form=FORM)
    body = _post("11월 17일 오전 10시에 박지훈 님과 설비 미팅으로 방문 신청할게요. 홍길동 010-1234-5678 한빛정밀입니다.")
    assert body["status"] == "pending_confirmation"
    assert "방문자: 홍*동" in body["answer"]
    assert "휴대폰: 010-****-5678" in body["answer"]
    assert "방문일시: 2026-11-17 (화) 10:00" in body["answer"]
    assert "홍길동" not in str(body) and not RAW_PHONE.search(str(body))
    assert _stored(isolated_store) == []


def test_확정하면_신청번호와_함께_저장하고_LLM_을_다시_부르지_않는다(use_model, isolated_store):
    """'확정' → status=submitted, 신청번호 발급, 저장 1건(원본 값), 확정 처리 중 LLM 호출 0회."""
    model = use_model(APPLY, form=FORM)
    first = _post("방문 신청합니다", None)
    calls_before = model.calls
    body = _post("확정", first["thread_id"])
    assert body["status"] == "submitted"
    assert "GM-20261112-0001" in body["answer"]
    assert model.calls == calls_before  # 확정 턴에는 에이전트 모델을 부르지 않는다
    stored = _stored(isolated_store)
    assert len(stored) == 1
    # 저장소에는 원본 값이 들어간다 (응답에는 마스킹된 값만 나간다)
    assert stored[0]["visitor_name"] == "홍길동" and stored[0]["phone"] == "010-1234-5678"
    assert stored[0]["request_id"] == "GM-20261112-0001"


def test_취소하면_저장하지_않는다(use_model, isolated_store):
    """'취소' → status=cancelled, 저장 0건."""
    use_model(APPLY, form=FORM)
    first = _post("방문 신청합니다")
    body = _post("취소", first["thread_id"])
    assert body["status"] == "cancelled"
    assert _stored(isolated_store) == []


def test_확정_취소가_아닌_입력은_다시_묻는다(use_model, isolated_store):
    """확정 대기 중 '응' 같은 다른 입력은 다시 묻고(저장 0건), 그 뒤 '확정' 하면 저장된다."""
    use_model(APPLY, form=FORM)
    first = _post("방문 신청합니다")
    again = _post("응 그렇게 해줘", first["thread_id"])
    assert again["status"] == "pending_confirmation"
    assert again["answer"].startswith("'확정' 또는 '취소' 로만 답해 주세요")
    assert _stored(isolated_store) == []
    done = _post("확정", first["thread_id"])
    assert done["status"] == "submitted"
    assert len(_stored(isolated_store)) == 1


def test_접수된_대화에서_LLM_이_다시_신청하려_해도_저장하지_않는다(use_model, isolated_store):
    """submitted 이후 LLM 이 create_visit_request 를 다시 부르면 도구가 같은 신청번호를 안내하고 저장 건수는 그대로다."""
    use_model(APPLY + [tool_call("create_visit_request", {}, "call-2")], form=FORM)
    first = _post("방문 신청합니다")
    _post("확정", first["thread_id"])
    body = _post("같은 내용으로 한 번 더 신청해 주세요", first["thread_id"])
    assert "GM-20261112-0001" in body["answer"]
    assert body["status"] == "submitted"
    assert len(_stored(isolated_store)) == 1


# ── 4단계: 중복 저장 방지 ──────────────────────────────────────────


def _submitted_thread(use_model) -> tuple[str, object]:
    """신청 → 확정 대기 상태의 대화를 만들고 (thread_id, 가짜 모델) 을 돌려준다. 튜플 반환은 Java 의 Pair 와 같다."""
    model = use_model(APPLY, form=FORM)
    return _post("방문 신청합니다")["thread_id"], model


def test_접수_후_확정을_다시_보내면_LLM_호출_없이_같은_신청번호를_안내한다(use_model, isolated_store):
    """순차 재확정(응답 실패 뒤 재시도 포함): LLM 호출 전 state 검사로 판별 → LLM 0회, 저장 1건 유지."""
    thread_id, model = _submitted_thread(use_model)
    _post("확정", thread_id)
    calls_before = model.calls
    body = _post("확정", thread_id)
    assert body["status"] == "submitted"
    assert "GM-20261112-0001" in body["answer"]
    assert body["trace"][0]["step"] == "중복 확정 확인"
    assert model.calls == calls_before
    assert len(_stored(isolated_store)) == 1


def test_겹쳐_온_확정_두_건은_저장_1건이고_둘_다_같은_신청번호를_받는다(use_model, isolated_store, monkeypatch):
    """버튼 두 번 누르기: 첫 요청을 저장 직전에 붙잡아 둔 채 두 번째 요청을 보내 겹침을 강제한다.

    대화별 잠금이 있으면 두 번째 요청은 첫 요청이 끝날 때까지 기다렸다가 '이미 접수됨' 으로 같은 번호를 받는다.
    잠금이 없으면 두 요청이 모두 확정 대기를 재개해 각각 저장하므로 2건이 되어 이 테스트가 실패한다.
    """
    thread_id, _ = _submitted_thread(use_model)
    # threading.Event: 스레드끼리 신호를 주고받는 깃발 (Java 의 CountDownLatch(1) 과 비슷). set() 으로 올리고 wait() 로 기다린다.
    entered, release = threading.Event(), threading.Event()
    original = tools._append_request
    save_calls = []  # 저장 본체가 불린 횟수를 세는 목록. 저장 파일 건수만 보면 '덮어쓰기로 1건처럼 보이는' 경우를 놓친다.

    def held_append(fields):
        """저장 본체에 들어오면 '들어왔음' 신호를 보내고, 테스트가 풀어 줄 때까지 기다린 뒤 원래 저장을 한다."""
        save_calls.append(1)
        entered.set()
        release.wait(5)
        return original(fields)

    # monkeypatch.setattr: 테스트 동안만 tools._append_request 를 바꿔 끼운다 (Mockito 의 spy/doAnswer 와 비슷)
    monkeypatch.setattr(tools, "_append_request", held_append)
    responses = {}

    def confirm(key: str) -> None:
        """'확정' 을 보내고 응답을 responses[key] 에 담는다 (스레드에서 실행할 함수)."""
        responses[key] = client.post("/query", json={"question": "확정", "thread_id": thread_id})

    started = time.monotonic()
    first = threading.Thread(target=confirm, args=("first",))
    first.start()
    assert entered.wait(5), "첫 요청이 저장 단계까지 오지 못했다"
    second = threading.Thread(target=confirm, args=("second",))
    second.start()
    second.join(0.5)  # 0.5초 기다려도
    assert second.is_alive(), "두 번째 요청이 첫 요청을 기다리지 않고 끝났다 (잠금이 없음)"
    release.set()  # 첫 요청을 풀어 준다
    first.join(10)
    second.join(10)
    assert time.monotonic() - started < 10  # 두 요청 모두 10초 안에 끝난다 (Seed AC3)

    bodies = [responses["first"].json(), responses["second"].json()]
    assert all(r.status_code == 200 for r in responses.values())
    assert [b["status"] for b in bodies] == ["submitted", "submitted"]
    assert all("GM-20261112-0001" in b["answer"] for b in bodies)
    assert len(save_calls) == 1  # 저장 시도 자체가 한 번뿐이다
    assert len(_stored(isolated_store)) == 1


def test_서로_다른_대화가_동시에_저장해도_2건이고_신청번호가_다르다(isolated_store, monkeypatch):
    """저장소 잠금: 두 저장이 '읽기 → 번호 정하기 → 쓰기' 사이에서 겹치게 읽기를 느리게 만들어도 기록이 사라지거나 번호가 겹치지 않는다."""
    original = tools._load_json

    def slow_load(path):
        """파일을 읽은 뒤 0.2초 쉰다. 잠금이 없으면 이 사이에 다른 스레드도 같은 내용을 읽어 번호가 겹친다."""
        data = original(path)
        time.sleep(0.2)
        return data

    monkeypatch.setattr(tools, "_load_json", slow_load)
    ids = []
    # 리스트 컴프리헨션으로 스레드 2개를 만든다. lambda: ... 는 인자 없는 익명 함수 (Java 의 () -> ...)
    workers = [threading.Thread(target=lambda: ids.append(tools.save_request(FORM))) for _ in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(10)
    assert sorted(ids) == ["GM-20261112-0001", "GM-20261112-0002"]
    assert len(_stored(isolated_store)) == 2


# ── 검증 오류 (시나리오 10~15) ─────────────────────────────────────


def _validate(**changes) -> list[str]:
    """FORM 에서 일부 항목만 바꿔 검증한 오류 목록. **changes 는 '이름=값' 인자를 dict 로 받는다 (Java 의 Map<String,Object>)."""
    return tools.validate_request({**FORM, **changes})


def test_정상_신청은_검증을_통과한다():
    """평일, 기준일 이후, 형식이 맞고 명단에 있는 피방문자면 오류가 없다."""
    assert _validate() == []


def test_주말_과거_당일_방문은_거부한다():
    """시나리오 11(토요일), 12(과거), 13(당일)."""
    assert any("주말" in e for e in _validate(visit_datetime="2026-11-14T11:00"))
    assert any("지난 날짜" in e for e in _validate(visit_datetime="2026-11-10T10:00"))
    assert any("당일" in e for e in _validate(visit_datetime="2026-11-12T15:00"))


def test_휴대폰번호_형식과_없는_피방문자는_거부한다():
    """시나리오 14(010-123-45), 10(최가상)."""
    assert any("휴대폰번호 형식" in e for e in _validate(phone="010-123-45"))
    assert any("찾을 수 없습니다" in e for e in _validate(host="최가상"))


def test_빠진_필수_항목을_알려준다():
    """시나리오 15처럼 정보가 부족하면 빠진 항목 이름을 알려준다."""
    errors = tools.validate_request({"visit_datetime": "2026-11-17T10:00"})
    assert any("방문자" in e and "휴대폰" in e for e in errors)


def test_검증_오류면_확정_대기로_가지_않고_저장하지_않는다(use_model, isolated_store):
    """주말 신청은 도구가 오류 안내로 바로 답하고(return_direct), status 는 collecting, 저장 0건."""
    use_model(APPLY, form={**FORM, "visit_datetime": "2026-11-14T11:00"})
    body = _post("11월 14일 토요일 오전 11시 방문 신청")
    assert body["status"] == "collecting"
    assert "주말" in body["answer"]
    assert _stored(isolated_store) == []


def test_피방문자_확인_도구():
    """find_host: 명단에 있으면 부서와 함께 확인, 없으면 다시 확인 요청. 호칭 '님' 은 떼고 찾는다."""
    assert "설비기술팀" in tools.find_host.invoke({"name": "박지훈 님"})
    assert "명단에 없습니다" in tools.find_host.invoke({"name": "최가상"})


# ── 오류와 저장 ────────────────────────────────────────────────────


def test_신청_중_Bedrock_타임아웃이면_503_이고_저장하지_않는다(use_model, isolated_store):
    """모델 호출이 타임아웃되면 503, 저장 0건."""
    use_model([ReadTimeoutError(endpoint_url="https://bedrock")], form=FORM)
    res = client.post("/query", json={"question": "방문 신청합니다"})
    assert res.status_code == 503
    assert _stored(isolated_store) == []


def test_신청번호는_같은_날짜_안에서_일련번호로_늘어난다(isolated_store):
    """같은 기준일에 두 번 저장하면 0001, 0002 순서로 발급된다."""
    assert tools.save_request(FORM) == "GM-20261112-0001"
    assert tools.save_request(FORM) == "GM-20261112-0002"
